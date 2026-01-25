import re

def main():
    with open('src/backend/vulkan/backend.rs', 'r', encoding='utf-8') as f:
        c = f.read()

    # Fix builder patterns
    c = c.replace('.ctx.command_pool(', '.command_pool(')
    c = c.replace('.ctx.descriptor_pool(', '.descriptor_pool(')

    # Ensure imports are present
    if 'use super::swapchain::VulkanSurfaceContext;' not in c:
        c = c.replace('use super::context::VulkanContext;', 'use super::context::VulkanContext;\nuse super::swapchain::VulkanSurfaceContext;')

    with open('src/backend/vulkan/backend.rs', 'w', encoding='utf-8') as f:
        f.write(c)

if __name__ == '__main__':
    main()
