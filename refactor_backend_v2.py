import re

def main():
    with open('src/backend/vulkan/backend.rs', 'r', encoding='utf-8') as f:
        c = f.read()

    # 1. Update handle accessors
    # Surface/Swapchain handles -> surface_ctx
    surface_handles = ['surface', 'swapchain', 'swapchain_images', 'swapchain_image_views', 'framebuffers', 'surface_loader', 'swapchain_loader']
    for h in surface_handles:
        # Match .h but NOT .surface_ctx.h
        c = re.sub(r'(?<!surface_ctx)\.(' + h + r')\b', r'.surface_ctx.\1', c)
    
    # Pool handles -> ctx (command_pool and descriptor_pool)
    pool_handles = ['command_pool', 'descriptor_pool']
    for h in pool_handles:
        c = re.sub(r'(?<!ctx)\.(' + h + r')\b', r'.ctx.\1', c)

    # 2. Fix Width/Height -> surface_ctx.extent
    # Be careful not to replace struct field definitions or local variables in new() if we refactor it
    c = c.replace('self.width', 'self.surface_ctx.extent.width')
    c = c.replace('self.height', 'self.surface_ctx.extent.height')

    # 3. Refactor new() method - Surgical replacement
    # We'll replace the entire method body or use a template
    # Since it's huge, let's just fix the offending blocks
    
    # Fix seed_layout_info (temporary dropped while borrowed)
    # Search for the block creating seed_layout_info
    c = re.sub(
        r'let seed_layout_info = vk::PipelineLayoutCreateInfo::default\(\)\s*\.set_layouts\(&\[([a-zA-Z0-9_]+)\]\)\s*\.push_constant_ranges\(&\[([^\]]+)\]\);',
        r'let layouts = [\1];\n            let push_ranges = [\2];\n            let seed_layout_info = vk::PipelineLayoutCreateInfo::default()\n                .set_layouts(&layouts)\n                .push_constant_ranges(&push_ranges);',
        c
    )

    # 4. Refactor Drop implementation
    drop_impl = """impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            self.surface_ctx.destroy(&self.ctx.device);
            self.ctx.device.destroy_render_pass(self.render_pass, None);
            self.ctx.device.destroy_render_pass(self.render_pass_load, None);

            // Destroy other internal resources
            self.ctx.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.ctx.device.destroy_pipeline(self.pipeline, None);
            self.ctx.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            self.ctx.device.destroy_buffer(self.vertex_buffer, None);
            self.ctx.device.free_memory(self.vertex_memory, None);
            self.ctx.device.destroy_buffer(self.uniform_buffer, None);
            self.ctx.device.free_memory(self.uniform_memory, None);

            // ... (Add others if missing, but let's keep it clean) ...
            // self.ctx and self.instance are handled by VulkanContext's Drop
        }
    }
}"""
    c = re.sub(r'impl Drop for VulkanBackend \{.*?\}', drop_impl, c, flags=re.DOTALL)

    with open('src/backend/vulkan/backend.rs', 'w', encoding='utf-8') as f:
        f.write(c)

if __name__ == '__main__':
    main()
