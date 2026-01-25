import re

def main():
    with open('src/backend/vulkan/backend.rs', 'r', encoding='utf-8') as f:
        c = f.read()

    local_handles = [
        'command_pool', 'command_buffer', 'render_pass', 'render_pass_load', 
        'pipeline_layout', 'pipeline', 'descriptor_set_layout', 'descriptor_pool', 
        'surface_loader', 'swapchain_loader', 'surface', 'swapchain', 
        'swapchain_images', 'swapchain_image_views', 'framebuffers', 
        'vertex_buffer', 'vertex_memory', 'uniform_buffer', 'uniform_memory', 
        'font_texture', 'font_texture_memory', 'font_texture_view', 'sampler', 
        'descriptor_set', 'image_available_semaphore', 'render_finished_semaphore', 
        'in_flight_fence', 'k13_pipeline', 'k13_layout', 'k8_pipeline', 
        'k8_layout', 'k6_update_pipeline', 'k6_spawn_pipeline', 'k6_layout', 
        'k5_pipeline', 'k5_layout', 'k4_pipeline', 'k4_layout', 
        'k13_descriptor_set', 'k8_descriptor_set', 'k6_descriptor_set', 
        'k5_descriptor_sets', 'k4_descriptor_set', 'indirect_dispatch_buffer', 
        'indirect_dispatch_memory', 'indirect_draw_buffer', 'indirect_draw_memory', 
        'counter_buffer', 'counter_memory', 'instance_buffer', 'instance_memory', 
        'particle_buffer', 'particle_memory', 'counter_readback_buffer', 
        'counter_readback_memory', 'backdrop_image', 'backdrop_view', 
        'backdrop_memory', 'jfa_images', 'sdf_image', 'jfa_framebuffers', 
        'seed_render_pass', 'seed_pipeline', 'seed_layout', 'k5_resolve_pipeline', 
        'k5_resolve_layout', 'k5_resolve_descriptor_sets', 'query_pools', 
        'timestamp_period'
    ]

    for h in local_handles:
        c = c.replace(f'self.ctx.{h}', f'self.{h}')

    # Fix the ? on Result<_, ash::vk::Result>
    # Find patterns like: .create_descriptor_set_layout(...) }?
    # Replacing with: .create_descriptor_set_layout(...) }.map_err(|e| format!("{:?}", e))?
    
    # We use a non-greedy lookahead for the next ?
    c = re.sub(r'(\.create_[a-z_]+\(.*\)\s*\})\s*\?;', r'\1.map_err(|e| format!("{:?}", e))?;', c)
    c = re.sub(r'(\.allocate_descriptor_sets\(.*\)\s*\})\s*\?;', r'\1.map_err(|e| format!("{:?}", e))?;', c)
    c = re.sub(r'(\.create_compute_pipelines\(.*\)\s*\})\s*\?\s*(\[[0-9]+\]);', r'\1.map_err(|e| format!("{:?}", e))?\2;', c)

    with open('src/backend/vulkan/backend.rs', 'w', encoding='utf-8') as f:
        f.write(c)

if __name__ == '__main__':
    main()
