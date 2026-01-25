import re

def main():
    with open('src/backend/vulkan/backend.rs', 'r', encoding='utf-8') as f:
        c = f.read()

    # 1. Replace single-element temporary arrays with std::slice::from_ref
    # Pattern: .set_layouts(&[([a-zA-Z0-9_]+)])
    c = re.sub(r'\.set_layouts\(&\[([a-zA-Z0-9_]+)\]\)', r'.set_layouts(std::slice::from_ref(&\1))', c)
    c = re.sub(r'\.buffer_info\(&\[([a-zA-Z0-9_]+)\]\)', r'.buffer_info(std::slice::from_ref(&\1))', c)
    c = re.sub(r'\.image_info\(&\[([a-zA-Z0-9_]+)\]\)', r'.image_info(std::slice::from_ref(&\1))', c)
    c = re.sub(r'\.attachments\(&\[([a-zA-Z0-9_]+)\]\)', r'.attachments(std::slice::from_ref(&\1))', c)
    c = re.sub(r'\.push_constant_ranges\(&\[([a-zA-Z0-9_.]+)\]\)', r'.push_constant_ranges(std::slice::from_ref(&\1))', c)

    # 2. Fix multi-element temporary arrays in builders
    # We'll target the ones mentioned in the error log or common patterns
    # For k5_set_layout (line 952 approx)
    c = re.sub(
        r'let ([a-zA-Z0-9_]+) = vk::DescriptorSetAllocateInfo::default\(\)\s*\.descriptor_pool\(descriptor_pool\)\s*\.set_layouts\(&\[([^\]]+)\]\);',
        r'let layouts = [\2];\n        let \1 = vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts);',
        c
    )

    with open('src/backend/vulkan/backend.rs', 'w', encoding='utf-8') as f:
        f.write(c)

if __name__ == '__main__':
    main()
