use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use crate::core::Vec2;

pub struct AudioEngine {
    stream: Option<cpal::Stream>,
    state: Arc<Mutex<AudioState>>,
}

struct AudioState {
    voices: Vec<Voice>,
    listener_pos: Vec2,
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
    pub fn new() -> Self {
        let host = cpal::default_host();
        let device = host.default_output_device().expect("No audio device");
        let config = device.default_output_config().expect("No audio config");
        let sample_rate = config.sample_rate().0 as f32;

        let state = Arc::new(Mutex::new(AudioState {
            voices: Vec::new(),
            listener_pos: Vec2::ZERO,
            sample_rate,
            next_id: 1,
        }));

        let state_clone = state.clone();
        let channels = config.channels() as usize;

        let err_fn = |err| eprintln!("Audio error: {}", err);
        
        // We handle f32 by default.
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_output_stream(
                &config.into(),
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                     write_data(data, channels, &state_clone)
                },
                err_fn,
                None
            ),
            _ => panic!("Unsupported sample format (F32 expected for demo)"),
        }.expect("Stream build failed");

        stream.play().expect("Play failed");

        Self {
            stream: Some(stream),
            state,
        }
    }

    pub fn set_listener_pos(&self, pos: Vec2) {
        if let Ok(mut state) = self.state.lock() {
            state.listener_pos = pos;
        }
    }

    pub fn play_sound(&self, id: u64, sound_type: SoundType, pos: Vec2, volume: f32, pitch: f32, looping: bool) {
        if let Ok(mut state) = self.state.lock() {
            // Replace existing or add new
            if let Some(v) = state.voices.iter_mut().find(|v| v.id == id) {
                v.position = pos;
                v.volume = volume;
                v.pitch = pitch;
                v.playing = true;
                v.looping = looping;
                v.sample_cursor = 0.0;
            } else {
                state.voices.push(Voice {
                    id,
                    sound_type,
                    position: pos,
                    volume,
                    pitch,
                    looping,
                    sample_cursor: 0.0,
                    playing: true,
                    params: SoundParams { attack: 0.01, decay: 0.1, sustain: 0.8, release: 0.1, duration: 2.0 },
                    seed: id as u32,
                });
            }
        }
    }

    /// Fire-and-forget SFX playback (reuses finished voices)
    pub fn play_sfx(&self, sound_type: SoundType, volume: f32, pitch: f32) {
        if let Ok(mut state) = self.state.lock() {
            let mut reused_index = None;
            let seed = state.next_id as u32; // Simple seed
            let listener_pos = state.listener_pos; // Copy listener pos
            
            // Try to find a finished voice to recycle
            for (i, v) in state.voices.iter_mut().enumerate() {
                if !v.playing {
                    reused_index = Some(i);
                    break;
                }
            }

            if let Some(index) = reused_index {
                let v = &mut state.voices[index];
                v.sound_type = sound_type;
                v.volume = volume;
                v.pitch = pitch;
                v.playing = true;
                v.looping = false;
                v.sample_cursor = 0.0;
                v.position = listener_pos;
                v.seed = seed;
                v.params = SoundParams { attack: 0.01, decay: 0.1, sustain: 0.8, release: 0.1, duration: 1.0 };
            } else {
                let id = state.next_id;
                state.next_id += 1;
                state.voices.push(Voice {
                    id,
                    sound_type,
                    position: listener_pos,
                    volume,
                    pitch,
                    looping: false,
                    sample_cursor: 0.0,
                    playing: true,
                    params: SoundParams { attack: 0.01, decay: 0.1, sustain: 0.8, release: 0.1, duration: 1.0 },
                    seed: id as u32,
                });
            }
        }
    }
    
    pub fn update_voice(&self, id: u64, pos: Vec2, volume: f32) {
        if let Ok(mut state) = self.state.lock() {
            if let Some(v) = state.voices.iter_mut().find(|v| v.id == id) {
                v.position = pos;
                v.volume = volume;
            }
        }
    }
}

fn write_data(output: &mut [f32], channels: usize, state: &Arc<Mutex<AudioState>>) {
    // Fill with silence
    for sample in output.iter_mut() {
        *sample = 0.0;
    }

    if let Ok(mut state) = state.lock() {
        let sr = state.sample_rate;
        let listener = state.listener_pos;

        for voice in &mut state.voices {
            if !voice.playing { continue; }

            // Check duration
            let t = voice.sample_cursor / sr;
            if !voice.looping && t > voice.params.duration {
                voice.playing = false;
                continue;
            }

            // Spatialization
            let delta = voice.position - listener;
            let dist = (delta.x * delta.x + delta.y * delta.y).sqrt();
            
            // Attenuation (Linear falloff for demo)
            let max_dist = 500.0;
            let attenuation = if dist > max_dist { 0.0 } else { 1.0 - (dist / max_dist) };
            if attenuation <= 0.0 { continue; }

            // Panning: delta.x negative = left, positive = right
            let pan = (delta.x / 500.0).max(-1.0).min(1.0);
            
            // Stereo volumes (Linear)
            let vol_l = (1.0 - pan).min(1.0);
            let vol_r = (1.0 + pan).min(1.0);

            // Envelope (Basic Fade Out at end)
            let mut env = 1.0;
            if !voice.looping {
                let remaining = voice.params.duration - t;
                if remaining < voice.params.release {
                    env = remaining / voice.params.release;
                }
            }

            let base_vol = voice.volume * attenuation * 0.2 * env; 

            let frame_count = output.len() / channels;
            for i in 0..frame_count {
                let sample = generate_sample(voice, sr);
                
                // Writing to channels
                if channels >= 2 {
                    // Simple stereo
                    if let Some(s) = output.get_mut(i * channels) { *s += sample * base_vol * vol_l; }
                    if let Some(s) = output.get_mut(i * channels + 1) { *s += sample * base_vol * vol_r; }
                } else {
                    if let Some(s) = output.get_mut(i * channels) { *s += sample * base_vol; }
                }
            }
        }
    }
    
    // Soft clipping to prevent harsh digital distortion
    for sample in output.iter_mut() {
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
            // Simple LCG
            voice.seed = voice.seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (voice.seed as f32 / 4294967296.0) * 2.0 - 1.0
        },
    }
}
