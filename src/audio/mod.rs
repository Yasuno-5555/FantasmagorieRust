use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crate::core::Vec2;
use ringbuf::{HeapRb, Producer, Consumer};
use std::sync::Arc;

pub struct AudioEngine {
    _stream: cpal::Stream,
    command_producer: Producer<AudioCommand, Arc<HeapRb<AudioCommand>>>,
    telemetry_consumer: Consumer<AudioTelemetry, Arc<HeapRb<AudioTelemetry>>>,
    next_id: u64,
}

pub struct AudioTelemetry {
    pub cpu_usage: f32,
    pub peak_level: f32,
    pub active_voices: u32,
}

enum AudioCommand {
    PlaySound { id: u64, sound_type: SoundType, pos: Vec2, volume: f32, pitch: f32, looping: bool },
    PlaySFX { sound_type: SoundType, volume: f32, pitch: f32, seed: u32 },
    UpdateVoice { id: u64, pos: Vec2, volume: f32 },
    SetListenerPos(Vec2),
    StopAll,
}

const MAX_VOICES: usize = 32;

struct AudioState {
    voices: Vec<Voice>,
    listener_pos: Vec2,
    sample_index: u64,
    sample_rate: f32,
    next_id: u64,
}

struct Voice {
    id: u64,
    sound_type: SoundType,
    position: Vec2,
    volume: f32,
    pitch: f32,
    looping: bool,
    sample_cursor: f32, // Floating point for pitch
    playing: bool,
    params: SoundParams,
    seed: u32,
}

#[derive(Clone, Copy)]
pub enum SoundType {
    Sine(f32), // Frequency
    Square(f32),
    Noise,
}

#[derive(Clone, Copy)]
struct SoundParams {
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
    duration: f32, // For procedural
}

impl AudioEngine {
    pub fn new(sample_rate: f32) -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host.default_output_device().ok_or("No output device found")?;
        let config = device.default_output_config().map_err(|e| format!("No default config: {}", e))?;

        let command_rb = HeapRb::<AudioCommand>::new(1024);
        let (command_producer, mut command_consumer) = command_rb.split();

        let telemetry_rb = HeapRb::<AudioTelemetry>::new(128);
        let (mut telemetry_producer, telemetry_consumer) = telemetry_rb.split();

        let mut voices = Vec::with_capacity(MAX_VOICES);
        for _ in 0..MAX_VOICES {
            voices.push(Voice {
                id: 0,
                sound_type: SoundType::Sine(440.0),
                position: Vec2::ZERO,
                volume: 0.0,
                pitch: 1.0,
                playing: false,
                looping: false,
                sample_cursor: 0.0,
                params: SoundParams { attack: 0.01, decay: 0.1, sustain: 0.8, release: 0.1, duration: 1.0 },
                seed: 0,
            });
        }

        let mut state = AudioState {
            voices,
            listener_pos: Vec2::ZERO,
            sample_index: 0,
            sample_rate,
            next_id: 1,
        };
        
        let channels = config.channels() as usize;
        let err_fn = |err| eprintln!("Audio error: {}", err);
        
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_output_stream(
                &config.into(),
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let start = std::time::Instant::now();
                    let frame_count = data.len() / channels;
                    
                    // 1. Process pending commands
                    while let Some(cmd) = command_consumer.pop() {
                        process_command(&mut state, cmd);
                    }
                    
                    // 2. Clear buffer
                    for x in data.iter_mut() { *x = 0.0; }

                    // 3. Update sample index
                    state.sample_index += frame_count as u64;

                    // 4. Mix native voices
                    write_data(data, channels, &mut state);
                    
                    // 5. Send telemetry
                    let duration = start.elapsed();
                    let cpu_usage = duration.as_secs_f32() / (data.len() as f32 / (channels as f32 * sample_rate));
                    
                    // Update peak meter
                    let mut peak = 0.0f32;
                    for sample in data.iter() {
                        peak = peak.max(sample.abs());
                    }
                    
                    let _ = telemetry_producer.push(AudioTelemetry {
                        cpu_usage,
                        peak_level: peak,
                        active_voices: state.voices.iter().filter(|v| v.playing).count() as u32,
                    });
                },
                err_fn,
                None,
            ).map_err(|e| format!("Failed to build audio stream: {}", e))?,
            _ => return Err("Unsupported sample format".to_string()),
        };

        stream.play().map_err(|e| format!("Failed to start audio stream: {}", e))?;

        Ok(Self {
            _stream: stream,
            command_producer,
            telemetry_consumer,
            next_id: 1,
        })
    }

    pub fn set_listener_pos(&mut self, pos: Vec2) {
        let _ = self.command_producer.push(AudioCommand::SetListenerPos(pos));
    }

    pub fn play_sound(&mut self, id: u64, sound_type: SoundType, pos: Vec2, volume: f32, pitch: f32, looping: bool) {
        let _ = self.command_producer.push(AudioCommand::PlaySound { id, sound_type, pos, volume, pitch, looping });
    }

    pub fn play_sfx(&mut self, sound_type: SoundType, volume: f32, pitch: f32) {
        let seed = self.next_id as u32;
        self.next_id += 1;
        let _ = self.command_producer.push(AudioCommand::PlaySFX { sound_type, volume, pitch, seed });
    }
    
    pub fn update_voice(&mut self, id: u64, pos: Vec2, volume: f32) {
        let _ = self.command_producer.push(AudioCommand::UpdateVoice { id, pos, volume });
    }

    pub fn pop_telemetry(&mut self) -> Option<AudioTelemetry> {
        self.telemetry_consumer.pop()
    }
}

fn process_command(state: &mut AudioState, cmd: AudioCommand) {
    match cmd {
        AudioCommand::SetListenerPos(pos) => {
            state.listener_pos = pos;
        }
        AudioCommand::PlaySound { id, sound_type, pos, volume, pitch, looping } => {
            if let Some(v) = state.voices.iter_mut().find(|v| v.id == id) {
                v.sound_type = sound_type;
                v.position = pos;
                v.volume = volume;
                v.pitch = pitch;
                v.playing = true;
                v.looping = looping;
                v.sample_cursor = 0.0;
            } else {
                if let Some(v) = state.voices.iter_mut().find(|v| !v.playing) {
                    v.id = id;
                    v.sound_type = sound_type;
                    v.position = pos;
                    v.volume = volume;
                    v.pitch = pitch;
                    v.looping = looping;
                    v.sample_cursor = 0.0;
                    v.playing = true;
                    v.params = SoundParams { attack: 0.01, decay: 0.1, sustain: 0.8, release: 0.1, duration: 2.0 };
                    v.seed = id as u32;
                }
            }
        }
        AudioCommand::PlaySFX { sound_type, volume, pitch, seed } => {
            let listener_pos = state.listener_pos;
            let mut reused_index = None;
            for (i, v) in state.voices.iter_mut().enumerate() {
                if !v.playing {
                    reused_index = Some(i);
                    break;
                }
            }

            if let Some(index) = reused_index {
                let v = &mut state.voices[index];
                v.id = state.next_id; 
                state.next_id += 1;
                v.sound_type = sound_type;
                v.volume = volume;
                v.pitch = pitch;
                v.playing = true;
                v.looping = false;
                v.sample_cursor = 0.0;
                v.position = listener_pos;
                v.seed = seed;
                v.params = SoundParams { attack: 0.01, decay: 0.1, sustain: 0.8, release: 0.1, duration: 1.0 };
            }
        }
        AudioCommand::UpdateVoice { id, pos, volume } => {
            if let Some(v) = state.voices.iter_mut().find(|v| v.id == id) {
                v.position = pos;
                v.volume = volume;
            }
        }
        AudioCommand::StopAll => {
            for v in &mut state.voices { v.playing = false; }
        }
    }
}

fn write_data(data: &mut [f32], channels: usize, state: &mut AudioState) {
    let sr = state.sample_rate;
    let listener = state.listener_pos;

    for voice in &mut state.voices {
        if !voice.playing { continue; }

        let t = voice.sample_cursor / sr;
        if !voice.looping && t > voice.params.duration {
            voice.playing = false;
            continue;
        }

        let delta = voice.position - listener;
        let dist = (delta.x * delta.x + delta.y * delta.y).sqrt();
        let max_dist = 500.0;
        let attenuation = if dist > max_dist { 0.0 } else { 1.0 - (dist / max_dist) };
        if attenuation <= 0.0 { continue; }

        let pan = (delta.x / 500.0).max(-1.0).min(1.0);
        let vol_l = (1.0 - pan).min(1.0);
        let vol_r = (1.0 + pan).min(1.0);

        let mut env = 1.0;
        if !voice.looping {
            let remaining = voice.params.duration - t;
            if remaining < voice.params.release {
                env = remaining / voice.params.release;
            }
        }

        let base_vol = voice.volume * attenuation * 0.2 * env; 
        let frame_count = data.len() / channels;
        for i in 0..frame_count {
            let sample = generate_sample(voice, sr);
            if channels >= 2 {
                if let Some(s) = data.get_mut(i * channels) { *s += sample * base_vol * vol_l; }
                if let Some(s) = data.get_mut(i * channels + 1) { *s += sample * base_vol * vol_r; }
            } else {
                if let Some(s) = data.get_mut(i * channels) { *s += sample * base_vol; }
            }
        }
    }
    
    for sample in data.iter_mut() {
        *sample = sample.tanh();
    }
}

fn generate_sample(voice: &mut Voice, sr: f32) -> f32 {
    voice.sample_cursor += voice.pitch;
    let t = voice.sample_cursor / sr;
    match voice.sound_type {
        SoundType::Sine(freq) => (t * freq * 2.0 * std::f32::consts::PI).sin(),
        SoundType::Square(freq) => if (t * freq * 2.0 * std::f32::consts::PI).sin() > 0.0 { 1.0 } else { -1.0 },
        SoundType::Noise => {
            voice.seed = voice.seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (voice.seed as f32 / 4294967296.0) * 2.0 - 1.0
        },
    }
}
