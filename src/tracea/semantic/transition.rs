/// 菴咲嶌・・hase・峨ｒ Z/NZ 蟾｡蝗樒ｾ､縺ｨ縺励※螳夂ｾｩ
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Phase(pub u32);

#[derive(Debug, Clone, Copy)]
pub enum SyncRequirement {
    None,
    /// cp.async.wait_group<N> 縺ｫ逶ｸ蠖・
    WaitAsyncLoad { stages_behind: u32 },
    /// __syncthreads() 逶ｸ蠖・
    Barrier,
}

#[derive(Debug, Clone)]
pub struct PhaseTransition {
    pub from: Phase,
    pub to: Phase,
    pub num_stages: u32,
}

impl PhaseTransition {
    pub fn new(from: u32, to: u32, num_stages: u32) -> Self {
        Self {
            from: Phase(from % num_stages),
            to: Phase(to % num_stages),
            num_stages,
        }
    }

    /// 谺｡縺ｮ繧ｹ繝・・繧ｸ縺ｸ騾ｲ繧縺溘ａ縺ｮ蜷梧悄蜻ｽ莉､繧堤ｮ怜・
    /// 繧ｹ繝・・繧ｸ髢薙・霍晞屬縺ｫ蝓ｺ縺･縺・※蜷梧悄縺ｮ驥阪∩繧呈ｱｺ螳・
    pub fn required_sync(&self) -> SyncRequirement {
        // 萓・ N繧ｹ繝・・繧ｸ繝代う繝励Λ繧､繝ｳ縺ｮ蝣ｴ蜷医・
        // 繝ｭ繝ｼ繝峨・螳御ｺ・ｾ・ｩ溘・ (num_stages - 2) 蛟句燕繧呈欠螳壹☆繧九・縺御ｸ闊ｬ逧・
        // (譛譁ｰ縺ｮcommit縺九ｉ謨ｰ縺医※縺・￥縺､蜑阪・繧ｰ繝ｫ繝ｼ繝励′螳御ｺ・＠縺ｦ縺・ｋ縺ｹ縺阪°)
        if self.num_stages > 1 {
            SyncRequirement::WaitAsyncLoad {
                stages_behind: (self.num_stages.saturating_sub(2)),
            }
        } else {
            SyncRequirement::Barrier
        }
    }
}
