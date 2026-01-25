use crate::core::Vec2;
use crate::backend::GraphicsBackend;
use crate::config::{EngineConfig, Profile, Pipeline};
use super::frame::{FrameDescription, RenderCommand};
use super::packet::DrawPacket;
use super::types::*;
use super::draw_builder::DrawBuilder;

/// The rendering boundary layer.
///
/// Translates high-level commands into GPU draw calls via the backend.
/// Behavior varies based on the engine profile:
/// - **Lite:** Forward rendering, call order = draw order.
/// - **Cinema:** Satsuei Pipeline with render graph (not yet implemented).
pub struct Renderer {
    backend: Box<dyn GraphicsBackend>,
    config: EngineConfig,
    profiler: crate::renderer::gpu::ProfilerRegistry,
}

impl Renderer {
    /// Create a new Renderer with the specified backend and configuration.
    ///
    /// # Panics
    /// Panics if `Profile::Cinema` is selected (not yet implemented - Stage 3+ target).
    pub fn new(backend: Box<dyn GraphicsBackend>, config: EngineConfig) -> Self {
        // The Fork: Profile determines the entire rendering strategy
        match config.profile {
            Profile::Lite => {
                // Current implementation: Forward rendering, immediate-ish API
                // This path is fully functional
            }
            Profile::Cinema => {
                // Future: Satsuei Pipeline with RenderGraph
                // Semantics Pass → Composite Pass → Lens Pass
                panic!(
                    "Profile::Cinema is not yet implemented.\n\
                     This is a Stage 3+ feature requiring:\n\
                     - HDR (RGBA16F) color space\n\
                     - Render Graph infrastructure\n\
                     - Satsuei Pipeline (Semantics/Composite/Lens passes)\n\
                     \n\
                     Use EngineConfig::lite() for now."
                );
            }
        }

        // Validate pipeline matches profile
        match (&config.profile, &config.pipeline) {
            (Profile::Lite, Pipeline::Satsuei) => {
                panic!(
                    "Invalid configuration: Satsuei pipeline requires Profile::Cinema.\n\
                     Use Pipeline::Forward with Profile::Lite."
                );
            }
            _ => {}
        }

        Self { 
            backend, 
            config, 
            profiler: crate::renderer::gpu::ProfilerRegistry::new(),
        }
    }

    /// Create a Renderer with Lite profile (convenience constructor).
    pub fn new_lite(backend: Box<dyn GraphicsBackend>) -> Self {
        Self::new(backend, EngineConfig::lite())
    }

    /// Get the current engine configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Check if running in Lite mode.
    #[inline]
    pub fn is_lite(&self) -> bool {
        self.config.is_lite()
    }

    /// Execution statistics for GPU kernels (K11)
    pub fn stats(&self) -> &crate::renderer::gpu::KernelStats {
        &self.profiler.stats
    }

    pub fn begin_frame(&mut self) -> FrameContext {
        FrameContext::new()
    }

    /// Submits the frame.
    /// In Lite mode, this transmutes RenderCommands into DrawCommands for the backend.
    pub fn end_frame(&mut self, frame: FrameContext, width: u32, height: u32) {
        let description = frame.finish();
        
        // The Fork: Profile determines how we process the frame
        match self.config.profile {
            Profile::Lite => {
                // --- Lite Path: Forward Rendering ---
                // Transmute RenderCommands (Meaning) -> DrawCommands (Muscle)
                let mut dl = crate::draw::DrawList::new();
                self.transmute_to_drawlist(&description, &mut dl);

                // Execute via immediate-mode backend interface
                self.backend.render(&dl, width, height);
                self.backend.present();

                // K11: Update Profiler Stats
                if let Some((timestamps, period)) = self.backend.get_profiling_results() {
                    let total = timestamps[1].saturating_sub(timestamps[0]);
                    // period is in ns
                    self.profiler.stats.frame_time_ns = (total as f32 * period) as u64;
                }
            }
            Profile::Cinema => {
                // --- Cinema Path: Satsuei Pipeline ---
                // Future: let packets = tracea::optimize(description);
                let packets = self.mock_tracea_optimize(&description);
                self.backend.submit(&packets);
                self.backend.present();
            }
        }
    }

    /// Convert high-level RenderCommands into backend-friendly DrawCommands.
    fn transmute_to_drawlist(&self, frame: &FrameDescription, dl: &mut crate::draw::DrawList) {
        use crate::draw::DrawCommand;
        let mut active_camera: Option<&super::camera::Camera> = None;

        for cmd in &frame.commands {
            match cmd {
                RenderCommand::BeginWorld(cam) => {
                    active_camera = Some(cam);
                }
                RenderCommand::EndWorld => {
                    active_camera = None;
                }
                RenderCommand::DrawQuad { rect, color } => {
                    let (pos, size) = if let Some(cam) = active_camera {
                        let p = cam.world_to_screen(rect.pos());
                        let s = rect.size() * cam.zoom;
                        (Vec2::new(p.x - s.x * 0.5, p.y - s.y * 0.5), s)
                    } else {
                        (rect.pos(), rect.size())
                    };
                    dl.add_rect(pos, size, *color);
                }
                RenderCommand::DrawTexturedQuad { rect, uv, texture: t } => {
                    let (pos, size) = if let Some(cam) = active_camera {
                        let p = cam.world_to_screen(rect.pos());
                        let s = rect.size() * cam.zoom;
                        (Vec2::new(p.x - s.x * 0.5, p.y - s.y * 0.5), s)
                    } else {
                        (rect.pos(), rect.size())
                    };
                    // Use the handle's value as u64 for DrawList
                    dl.add_image(pos, size, t.0 as u64, [uv.u, uv.v, uv.u + uv.w, uv.v + uv.h], crate::core::ColorF::white());
                }
                RenderCommand::DrawShape {
                    rect,
                    color,
                    radii,
                    border,
                    glow,
                    elevation,
                    is_squircle,
                    morph,
                } => {
                    let (bw, bc) = border.map(|b| (b.width, b.color)).unwrap_or((0.0, crate::core::ColorF::transparent()));
                    let (gs, gc) = glow.map(|g| (g.strength, g.color)).unwrap_or((0.0, crate::core::ColorF::transparent()));

                    dl.add_rect_ex(
                        rect.pos(),
                        rect.size(),
                        radii.as_array(),
                        *color,
                        *elevation,
                        *is_squircle,
                        bw,
                        bc,
                        crate::core::Vec2::ZERO,
                        gs,
                        gc,
                    );
                }
                RenderCommand::SetScissor(rect) => {
                    dl.push_clip(rect.pos(), rect.size());
                }
                // ... handle other commands like transformations or mesh draws ...
                _ => {}
            }
        }
    }

    // Temporary mock for Tracea optimization logic
    fn mock_tracea_optimize(&self, frame: &FrameDescription) -> Vec<DrawPacket> {
        // Simple 1-to-1 mapping for testing (very inefficient, but valid flow)
        let mut packets = Vec::new();
        
        // In reality, Tracea would:
        // 1. Analyze frame.commands
        // 2. Sort by pipeline/texture
        // 3. Batch Quads
        // 4. Generate DrawPackets
        
        // Just creating a dummy packet to prove it works
        if !frame.commands.is_empty() {
            packets.push(DrawPacket {
                pipeline: PipelineHandle(0),
                vertex_buffer: BufferHandle(0),
                index_buffer: None,
                descriptor_set: DescriptorSetHandle(0),
                draw_range: DrawRange {
                    start: 0,
                    count: frame.commands.len() as u32 * 6, // Fake count
                    vertex_offset: 0,
                },
            });
        }
        
        packets
    }
}

pub struct FrameContext {
    description: FrameDescription,
    current_transform: Transform2D,
    current_scissor: Rect,
}

impl FrameContext {
    fn new() -> Self {
        Self {
            description: FrameDescription::new(),
            current_transform: Transform2D::identity(),
            current_scissor: Rect::new(0.0, 0.0, 8192.0, 8192.0),
        }
    }

    fn finish(self) -> FrameDescription {
        self.description
    }

    /// Internal helper to record a command.
    pub(crate) fn push_command(&mut self, cmd: RenderCommand) {
        self.description.commands.push(cmd);
    }

    // --- Modern Drawing API ---

    /// Start building a high-feature SDF shape (rounded rect, circle, border, etc).
    ///
    /// # Example
    /// ```rust
    /// frame.draw(Rect::new(10.0, 10.0, 100.0, 40.0), Color::WHITE)
    ///     .rounded(8.0)
    ///     .border(2.0, Color::BLACK)
    ///     .submit();
    /// ```
    pub fn draw(&mut self, rect: Rect, color: Color) -> DrawBuilder<'_> {
        DrawBuilder::new(self, rect, color)
    }

    // --- Immediate-ish API ---

    pub fn set_pipeline(&mut self, pipeline: PipelineHandle) {
        self.description.commands.push(RenderCommand::SetPipeline(pipeline));
    }

    pub fn set_texture(&mut self, slot: u32, handle: TextureHandle) {
        self.description.commands.push(RenderCommand::SetTexture { slot, handle });
    }

    pub fn set_transform(&mut self, transform: Transform2D) {
        self.current_transform = transform;
        self.description.commands.push(RenderCommand::SetTransform(transform));
    }

    pub fn set_scissor(&mut self, rect: Rect) {
        self.current_scissor = rect;
        self.description.commands.push(RenderCommand::SetScissor(rect));
    }

    pub fn draw_quad(&mut self, rect: Rect, color: Color) {
        self.description.commands.push(RenderCommand::DrawQuad { rect, color });
    }

    pub fn draw_textured_quad(&mut self, rect: Rect, uv: UVRect, texture: TextureHandle) {
        self.description.commands.push(RenderCommand::DrawTexturedQuad { rect, uv, texture });
    }

    pub fn draw_mesh(&mut self, mesh: MeshHandle) {
        self.description.commands.push(RenderCommand::DrawMesh(mesh));
    }

    // --- Game Engine API ---

    pub fn begin_world(&mut self, camera: &super::camera::Camera) {
        self.description.commands.push(RenderCommand::BeginWorld(camera.clone()));
    }

    pub fn end_world(&mut self) {
        self.description.commands.push(RenderCommand::EndWorld);
    }
}
