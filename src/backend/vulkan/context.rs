use ash::vk;
use std::sync::Arc;

/// Centralized Vulkan context to be shared between renderer and compute (Tracea)
pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub graphics_family: u32,
    pub compute_queue: vk::Queue,
    pub compute_family: u32,
    pub command_pool: vk::CommandPool,
    pub descriptor_pool: vk::DescriptorPool,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
}

impl std::fmt::Debug for VulkanContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanContext")
            .field("physical_device", &self.physical_device)
            .field("graphics_family", &self.graphics_family)
            .field("compute_family", &self.compute_family)
            .finish()
    }
}

impl VulkanContext {
    /// Create a new Vulkan context from scratch
    pub unsafe fn new(
        _hinstance: *mut std::ffi::c_void,
        _hwnd: *mut std::ffi::c_void,
    ) -> Result<Arc<Self>, String> {
        eprintln!("[VulkanContext] Starting initialization...");
        let entry = ash::Entry::load().map_err(|e| format!("Failed to load Vulkan: {:?}", e))?;
        eprintln!("[VulkanContext] Entry loaded");

        let app_name = std::ffi::CString::new("Fantasmagorie").unwrap();
        let engine_name = std::ffi::CString::new("Fantasmagorie Engine").unwrap();

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_2);

        let setup_validation = true; // Enable for debugging
        let layers = if setup_validation {
            vec![b"VK_LAYER_KHRONOS_validation\0".as_ptr() as *const i8]
        } else {
            vec![]
        };
        
        let extension_names = [
            ash::khr::surface::NAME.as_ptr(),
            ash::khr::win32_surface::NAME.as_ptr(),
        ];
        
        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);
            
        if setup_validation {
            create_info = create_info.enabled_layer_names(&layers);
        }

        let instance = entry
            .create_instance(&create_info, None)
            .map_err(|e| format!("Failed to create Vulkan instance: {:?}", e))?;
        eprintln!("[VulkanContext] Instance created");

        // Physical Device selection
        let pdevices = instance
            .enumerate_physical_devices()
            .map_err(|e| format!("Failed to enumerate physical devices: {:?}", e))?;
        for (i, p) in pdevices.iter().enumerate() {
            let props = instance.get_physical_device_properties(*p);
            let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy() };
            eprintln!("[VulkanContext] Physical device [{}]: {}", i, name);
        }
        let physical_device = pdevices
            .into_iter()
            .find(|&pd| {
                let props = instance.get_physical_device_properties(pd);
                props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                    || props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
            })
            .ok_or("No suitable GPU found")?;
        eprintln!("[VulkanContext] Physical device selected");

        // Queue families
        let queue_families = instance.get_physical_device_queue_family_properties(physical_device);
        
        let graphics_family = queue_families
            .iter()
            .position(|qf| qf.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or("No graphics queue family found")? as u32;

        let compute_family = queue_families
            .iter()
            .position(|qf| {
                qf.queue_flags.contains(vk::QueueFlags::COMPUTE) 
                && !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            })
            .or_else(|| {
                queue_families.iter().position(|qf| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
            })
            .ok_or("No compute queue family found")? as u32;

        // Logical Device
        let queue_priorities = [1.0f32];
        let mut queue_create_infos = vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(graphics_family)
                .queue_priorities(&queue_priorities),
        ];

        if compute_family != graphics_family {
            queue_create_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(compute_family)
                    .queue_priorities(&queue_priorities),
            );
        }

        let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions);

        let device = instance
            .create_device(physical_device, &device_create_info, None)
            .map_err(|e| format!("Failed to create logical device: {:?}", e))?;
        eprintln!("[VulkanContext] Logical device created");

        let graphics_queue = device.get_device_queue(graphics_family, 0);
        let compute_queue = device.get_device_queue(compute_family, 0);

        // Persistent Pools
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(graphics_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = device.create_command_pool(&pool_info, None).map_err(|e| format!("{:?}", e))?;

        let pool_sizes = [
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(100),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(100),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(100),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::SAMPLER).descriptor_count(100),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(100),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(100),
        ];
        let desc_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(500)
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let descriptor_pool = device.create_descriptor_pool(&desc_pool_info, None).map_err(|e| format!("{:?}", e))?;
        eprintln!("[VulkanContext] Context initialized successfully");

        let physical_device_properties = instance.get_physical_device_properties(physical_device);
        
        Ok(Arc::new(Self {
            entry,
            instance,
            physical_device,
            device,
            graphics_queue,
            graphics_family,
            compute_queue,
            compute_family,
            command_pool,
            descriptor_pool,
            physical_device_properties,
        }))
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
