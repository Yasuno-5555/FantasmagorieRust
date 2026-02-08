use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioEmitter {
    pub params_id: u64, // 0=Sine440, 1=Noise, 2=Pulse
    pub volume: f32,
    pub pitch: f32,
    pub looping: bool,
    pub playing: bool,
}

impl Default for AudioEmitter {
    fn default() -> Self {
        Self {
            params_id: 0,
            volume: 1.0,
            pitch: 1.0,
            looping: true,
            playing: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct AudioListener {
    pub active: bool,
}
