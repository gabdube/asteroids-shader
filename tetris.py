import sys
#sys.stdout = open('run_log.txt', 'w')
#sys.stderr = open('run_err_log.txt', 'w')
sys.path.append('./libs')

import time
import platform
from enum import IntFlag
from ctypes import byref, cast, pointer, sizeof, memset, c_uint32, c_uint16, c_uint8, c_float, c_double, c_char_p, c_uint64, c_size_t, c_void_p, Structure, POINTER
from collections import namedtuple

from system import Window, events
from vulkan import vk

DEBUG = True

#
# HELPERS METHODS
#

def array(type_, count, data=None):
    # Makes ctypes arrays a lot clearer
    array_type = type_*count
    if data is None:
        return array_type
    else:
        return array_type(*data)


def array_pointer(value, t=None):
    return cast(value, POINTER(t or value._type_))


def call_vk(fn, item_type, info, parent=None):
    # Wrapper for vulkan functions of this style:
    # - vkCreateStuff(stuff_info, allocation_callbacks, ref_to_stuff)
    # - vkCreateStuff(parent, stuff_info, allocation_callbacks, ref_to_stuff)

    item = item_type(0)
    if parent is not None:
        result = fn(parent, byref(info), None, byref(item))
    else:
        result = fn(byref(info), None, byref(item))

    if result != vk.SUCCESS:
        raise RuntimeError(f'Call to vulkan failed. Error code: 0x{result:X}')

    return item


def enumerate_vk(fn, parent, item, parent2=None):
    # Wrapper for vulkan functions of this style: vkEnumerateStuff(parent, count, objects)
    count = c_uint32(0)

    if parent2 is None:
        result = fn(parent, byref(count), None)
    else:
        result = fn(parent, parent2, byref(count), None)


    if result is not None and result != vk.SUCCESS:
        raise RuntimeError(f'Call to vulkan failed. Error code: 0x{result:X}')
    elif count.value == 0:
        raise RuntimeError(f'Call to vulkan function to get count returned 0')

    buf = array(item, count.value, ())

    if parent2 is None:
        result = fn(parent, byref(count), array_pointer(buf))
    else:
        result = fn(parent, parent2, byref(count), array_pointer(buf))

    if result is not None and result != vk.SUCCESS:
        raise RuntimeError(f'Call to vulkan failed. Error code: 0x{result:X}')

    # This condition make sure that vulkan handles stays in their ctypes object
    if item is c_size_t or item is c_uint64:
        return tuple(item(i) for i in buf)
    else:
        return buf


def cache_memory_properties(tetris):
    mproperties = vk.PhysicalDeviceMemoryProperties()
    tetris.GetPhysicalDeviceMemoryProperties(tetris.physical_device, byref(mproperties))

    memory_types = mproperties.memory_types[0:mproperties.memory_type_count]
    memory_heaps = mproperties.memory_heaps[0:mproperties.memory_heap_count]

    device_flag = vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    device_only_type = next((i for (i, t) in enumerate(memory_types) if t.property_flags == device_flag), None)
    device_type = next((i for (i, t) in enumerate(memory_types) if IntFlag(device_flag) in IntFlag(t.property_flags)), None)
    device_type_index = device_only_type or device_type
    if device_type_index is None:
        raise RuntimeError("Could not find device memory type")

    host_flags = vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.MEMORY_PROPERTY_HOST_COHERENT_BIT
    host_only_type = next((i for (i, t) in enumerate(memory_types) if t.property_flags == host_flags), None)
    host_type = next((i for (i, t) in enumerate(memory_types) if IntFlag(host_flags) in IntFlag(t.property_flags)), None)
    host_type_index = host_only_type or host_type
    if host_type_index is None:
        raise RuntimeError("Could not find host memory type")

    tetris.memory_types = memory_types
    tetris.memory_heaps = memory_heaps
    tetris.device_type_index = device_type_index
    tetris.host_type_index = host_type_index


def memory_type_index(tetris, flags):
    if not hasattr(tetris, 'device_type_index'):
        cache_memory_properties(tetris)

    flags = IntFlag(flags)
    device_flags = IntFlag(vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    if device_flags in flags:
        return tetris.device_type_index

    return tetris.host_type_index
    

def align(offset, align):
    return (offset + (align - 1)) & -align


def buffer_params(size, usage):
    return vk.BufferCreateInfo(
        type = vk.STRUCTURE_TYPE_BUFFER_CREATE_INFO, 
        next = None, 
        flags = 0,
        size = size,
        usage = usage,
        sharing_mode = vk.SHARING_MODE_EXCLUSIVE,
        queue_family_index_count = 0,
        queue_family_indices = None
    )


def create_buffer(tetris, info):
    return call_vk(tetris.CreateBuffer, vk.DeviceMemory, info, tetris.device)


def physical_device_surface_capabilities(tetris):
    caps = vk.SurfaceCapabilitiesKHR()
    result = tetris.GetPhysicalDeviceSurfaceCapabilitiesKHR(tetris.physical_device, tetris.surface, byref(caps))
    if result != vk.SUCCESS:
        raise RuntimeError(f"Failed to get surface capabilities: 0x{result:X}")

    return caps



#
# SYSTEM setup
#

def create_window(tetris):
    tetris.window = Window(width=800, height=800)


#
# BASIC VULKAN SETUP (stuff you probably won't care about)
#

def available_layers():
    layers_count = c_uint32(0)

    result = vk.EnumerateInstanceLayerProperties(byref(layers_count), None)
    if result != vk.SUCCESS:
        raise RuntimeError('Failed to find the instance layers. Error code: 0x{:X}'.format(result))

    layers = array(vk.LayerProperties, layers_count.value, ())

    result = vk.EnumerateInstanceLayerProperties(byref(layers_count), array_pointer(layers))
    if result != vk.SUCCESS:
        raise RuntimeError('Failed to find the instance layers. Error code: 0x{:X}'.format(result))

    return [layer.layer_name.decode('utf-8') for layer in layers]


def load_instance_functions(tetris, instance):
    for name, function in vk.load_functions(instance, vk.InstanceFunctions, vk.GetInstanceProcAddr):
        setattr(tetris, name, function)


def create_instance(tetris):
    system_surface_extensions = {"Windows": "VK_KHR_win32_surface", "Linux": "VK_KHR_xcb_surface"}
    system_surface_extension = system_surface_extensions.get(platform.system())
    if system_surface_extension is None:
        raise RuntimeError(f"Platform {platform.system()} is not supported")

    extensions = ["VK_KHR_surface", system_surface_extension]
    layers = []

    if DEBUG and "VK_LAYER_LUNARG_standard_validation" in available_layers():
        tetris.debug = True
        extensions.append("VK_EXT_debug_utils")
        layers.append("VK_LAYER_LUNARG_standard_validation")
    else:
        tetris.debug = False

    print("")
    print(f"Using extensions: {extensions}")
    print(f"Using layers: {layers}")
    
    _extensions = [e.encode('utf8') for e in extensions]
    _layers = [l.encode('utf8') for l in layers]
    extensions = array(c_char_p, len(_extensions), _extensions)
    layers = array(c_char_p, len(_layers), _layers)

    app_info = vk.ApplicationInfo(
        type=vk.STRUCTURE_TYPE_APPLICATION_INFO, next=None,
        application_name=b'TETRIS', application_version=0,
        engine_name=b'TETRIS', engine_version=0, api_version=vk.API_VERSION_1_0
    )

    create_info = vk.InstanceCreateInfo(
        type=vk.STRUCTURE_TYPE_INSTANCE_CREATE_INFO, next=None, flags=0,
        application_info=pointer(app_info),

        enabled_layer_count=len(_layers),
        enabled_layer_names=array_pointer(layers),

        enabled_extension_count=len(_extensions),
        enabled_extension_names=array_pointer(extensions)
    )

    tetris.instance = call_vk(vk.CreateInstance, vk.Instance, create_info)
    load_instance_functions(tetris, tetris.instance)    # We store all function pointers in the object


def create_debug_utils(tetris):
    if not tetris.debug:
        return

    #
    # Types
    #

    class MessageSeverity(IntFlag):
        Verbose = vk.DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
        Information = vk.DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT
        Warning = vk.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
        Error = vk.DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT

    class MessageType(IntFlag):
        General = vk.DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT 
        Performance = vk.DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
        Validation = vk.DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT

    def format_debug(message_severity, message_type, callback_data, user_data):
        message_severity = MessageSeverity(message_severity).name
        message_type = MessageType(message_type).name
        
        data = callback_data.contents
        message = data.message[::].decode()
        full_message = f"{message_severity}/{message_type} -> {message}"
        print(full_message)
        
        return 0

    #
    # Actual code
    # 

    callback_fn = vk.FnDebugUtilsMessengerCallbackEXT(lambda *args: format_debug(*args))

    create_info = vk.DebugUtilsMessengerCreateInfoEXT(
        type = vk.STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        next = None,
        flags = 0,
        message_severity = vk.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | vk.DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        message_type =  vk.DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT | vk.DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
        user_callback = callback_fn,
        user_data = None
    )

    tetris.callback_fn = callback_fn  # Must not be GC'd
    tetris.debug_utils = call_vk(tetris.CreateDebugUtilsMessengerEXT, vk.DebugUtilsMessengerEXT, create_info, parent=tetris.instance)


def create_surface(tetris):
    surface_info = create_surface = None
    window = tetris.window
    
    platform_name =  platform.system()
    if platform_name == "Windows":
        create_surface = tetris.CreateWin32SurfaceKHR
        surface_info = vk.Win32SurfaceCreateInfoKHR(
            type = vk.STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
            next = None,
            flags = 0,
            hinstance = window.module,
            hwnd = window.handle
        )
    elif platform_name == "Linux":
        create_surface = tetris.CreateXcbSurfaceKHR
        surface_info = vk.XcbSurfaceCreateInfoKHR(
            type = vk.STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
            next = None,
            flags = 0,
            connection = cast(window.connection, POINTER(vk.xcb_connection_t)),
            window = window.handle
        )

    tetris.surface = call_vk(create_surface, vk.SurfaceKHR, surface_info, parent=tetris.instance)


def available_device_extensions(tetris, physical_device):
    ext_count = c_uint32(0)

    result = tetris.EnumerateDeviceExtensionProperties(physical_device, None, byref(ext_count), None)
    if result != vk.SUCCESS:
        raise RuntimeError(f'Failed to find the device extensions. Error code: 0x{result:X}')

    extensions = array(vk.ExtensionProperties, ext_count.value, ())

    result = tetris.EnumerateDeviceExtensionProperties(physical_device, None, byref(ext_count), array_pointer(extensions))
    if result != vk.SUCCESS:
        raise RuntimeError(f'Failed to find the instance layers. Error code: 0x{result:X}')

    return [ext.extension_name.decode('utf-8') for ext in extensions]


def load_device_functions(tetris, device):
    for name, function in vk.load_functions(device, vk.DeviceFunctions, tetris.GetDeviceProcAddr):
        setattr(tetris, name, function)


def create_device(tetris):
    physical_devices = enumerate_vk(tetris.EnumeratePhysicalDevices, tetris.instance, vk.PhysicalDevice)
    physical_device = physical_devices[0]   # Good old, battle proven, "select the first physical device" technique

    # Properties
    device_props = vk.PhysicalDeviceProperties()
    tetris.GetPhysicalDeviceProperties(physical_device, byref(device_props))

    # Features
    supported_features = vk.PhysicalDeviceFeatures()
    tetris.GetPhysicalDeviceFeatures(physical_device, byref(supported_features))

    if supported_features.shader_storage_buffer_array_dynamic_indexing != vk.TRUE:
        raise RuntimeError("Storage buffer dynamic indexing is not supported")
    elif supported_features.shader_float64 != vk.TRUE:
        raise RuntimeError("Double in shaders are not supported")

    features = vk.PhysicalDeviceFeatures(
        shader_storage_buffer_array_dynamic_indexing = vk.TRUE,
        shader_float64 = vk.TRUE
    )

    # Extensions
    extensions = ["VK_KHR_swapchain"]
    valid_extensions = available_device_extensions(tetris, physical_device)
    
    if "VK_KHR_draw_indirect_count" in valid_extensions:
        extensions.append("VK_KHR_draw_indirect_count")      # Newish extension. Require pretty recent drivers.
    elif "VK_AMD_draw_indirect_count" in valid_extensions:
        extensions.append("VK_AMD_draw_indirect_count")      # Older extension available to pretty much all AMD hardware since forever. 
    else:
        raise RuntimeError("Draw indirect extension not support. It should be though. Make sure your vulkan API version is up to date (>= 1.1.73)")
    
    extensions_array = array(c_char_p, len(extensions), (e.encode('utf8') for e in extensions))

    print(f"Using device extensions: {extensions}")

    # Queues
    queue_families = tuple(enumerate(enumerate_vk(tetris.GetPhysicalDeviceQueueFamilyProperties, physical_device, vk.QueueFamilyProperties)))
    priorities = array(c_float, 1, (1.0,)) 
    priorities_ptr = array_pointer(priorities)
    
    # Queues - Graphics
    graphics_index, graphics = next((index, fam) for index, fam in queue_families if fam.queue_flags & vk.QUEUE_GRAPHICS_BIT == vk.QUEUE_GRAPHICS_BIT)
    graphics_queue_info = vk.DeviceQueueCreateInfo(
        type = vk.STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        next = None, 
        flags = 0,
        queue_family_index = graphics_index,
        queue_count = 1, 
        queue_priorities = priorities_ptr
    )

    surface_supported = vk.Bool32(0)
    result = tetris.GetPhysicalDeviceSurfaceSupportKHR(physical_device, graphics_index, tetris.surface, byref(surface_supported))
    if result != vk.SUCCESS:
        raise RuntimeError(f"Failed to get surface support: 0x{result:X}")
    elif surface_supported == 0:
        raise RuntimeError(f"A graphics queue without surface support????")

    # Actual device creation
    queue_create_infos = (graphics_queue_info,)
    queue_create_infos_array = array(vk.DeviceQueueCreateInfo, len(queue_create_infos), queue_create_infos)

    device_info = vk.DeviceCreateInfo(
        type=vk.STRUCTURE_TYPE_DEVICE_CREATE_INFO, next=None, flags=0,
        queue_create_info_count=len(queue_create_infos), queue_create_infos=array_pointer(queue_create_infos_array),
        enabled_layer_count=0, enabled_layer_names=None,
        enabled_extension_count=len(extensions), enabled_extension_names=array_pointer(extensions_array),
        enabled_features = pointer(features)
    )

    tetris.physical_device = physical_device
    tetris.device = device = call_vk(tetris.CreateDevice, vk.Device, device_info, parent=physical_device)
    load_device_functions(tetris, tetris.device)   # Again, we store all device function pointers in the object

    # Fetch the queues
    Queue = namedtuple("Queue", "handle family_index family_info")

    tetris.limits = device_props.limits
    tetris.graphics = Queue(vk.Queue(0), graphics_index, graphics)
    tetris.GetDeviceQueue(device, graphics_index, 0, byref(tetris.graphics.handle))

    print(f"\nPhysical device name: {device_props.device_name.decode('utf-8')}")


def create_swapchain(tetris):
    physical_device, device, surface = tetris.physical_device, tetris.device, tetris.surface
    old_swapchain = getattr(tetris, 'swapchain', 0)
    
    # Swapchain default Setup
    caps = physical_device_surface_capabilities(tetris)
    surface_formats = enumerate_vk(tetris.GetPhysicalDeviceSurfaceFormatsKHR, physical_device, vk.SurfaceFormatKHR, parent2=surface)
    present_modes = enumerate_vk(tetris.GetPhysicalDeviceSurfacePresentModesKHR, physical_device, c_uint32, parent2=surface)

    extent = caps.current_extent
    transform = caps.current_transform
    present_mode = vk.PRESENT_MODE_FIFO_KHR
    swp_format = vk.FORMAT_B8G8R8A8_UNORM
    swp_color_space = 0
    swp_count = 2
    swp_usage = vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.IMAGE_USAGE_TRANSFER_DST_BIT

    # Swapchain Format
    for fmt in surface_formats:
        if fmt.format == swp_format:
            swp_color_space = fmt.color_space
            break
    
    if swp_color_space is None:
        raise RuntimeError("Swapchain image format is not supported")

    # Swapchain extent
    if extent.width == -1 or extent.height == -1:
        width, height = window.dimensions()
        extent.width = width
        extent.height = height

    # Min image count
    if caps.max_image_count != 0 and caps.min_image_count > 2:
        raise RuntimeError("Minimum image count not met")

    # Present mode
    if vk.PRESENT_MODE_MAILBOX_KHR in present_modes:
        present_mode = vk.PRESENT_MODE_MAILBOX_KHR
    elif vk.PRESENT_MODE_IMMEDIATE_KHR in present_modes:
        present_mode = vk.PRESENT_MODE_IMMEDIATE_KHR

    # Default image transformation
    if IntFlag(vk.SURFACE_TRANSFORM_IDENTITY_BIT_KHR) in IntFlag(caps.supported_transforms):
        transform = vk.SURFACE_TRANSFORM_IDENTITY_BIT_KHR

    # Swapchain creation
    swapchain_info = vk.SwapchainCreateInfoKHR(
        type = vk.STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        next = None, 
        flags = 0,
        surface = surface,
        min_image_count = swp_count,
        image_format = swp_format,
        image_color_space = swp_color_space,
        image_extent = extent,
        image_array_layers = 1,
        image_usage = swp_usage,
        image_sharing_mode = vk.SHARING_MODE_EXCLUSIVE,
        queue_family_index_count = 0,
        queue_family_indices = None,
        pre_transform = transform,
        composite_alpha = vk.COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        present_mode = present_mode,
        clipped = False,
        old_swapchain = old_swapchain
    )

    tetris.extent = extent
    tetris.swapchain_format = swp_format
    tetris.swapchain = call_vk(tetris.CreateSwapchainKHR, vk.Device, swapchain_info, parent=device)

    if old_swapchain is not None:
        tetris.DestroySwapchainKHR(device, old_swapchain, None)


def create_swapchain_images(tetris):
    SwapchainImage = namedtuple("SwapchainImage", "image view")
    device, swapchain_format = tetris.device, tetris.swapchain_format

    swapchain_images = []
    raw_images = enumerate_vk(tetris.GetSwapchainImagesKHR, tetris.device, vk.Image, parent2=tetris.swapchain)

    for _, view in getattr(tetris, 'swapchain_images', ()):
        tetris.DestroyImageView(device, view, None)

    for image in raw_images:
        view_info = vk.ImageViewCreateInfo(
            type = vk.STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            next = None,
            flags = 0,
            image = image,
            view_type = vk.IMAGE_VIEW_TYPE_2D,
            format = swapchain_format,
            components = vk.ComponentMapping(),
            subresource_range = vk.ImageSubresourceRange(
                aspect_mask = vk.IMAGE_ASPECT_COLOR_BIT,
                base_mip_level = 0,
                level_count = 1,
                base_array_layer = 0,
                layer_count = 1,
            )
        )

        view = call_vk(tetris.CreateImageView, vk.ImageView, view_info, parent=device)
        swapchain_images.append(SwapchainImage(image, view))

    tetris.swapchain_images = tuple(swapchain_images)


def create_depth_stencil(tetris):
    physical_device = tetris.physical_device
    device, extent = tetris.device, tetris.extent
    width, height = extent.width, extent.height
    valid_depth_formats = (vk.FORMAT_D32_SFLOAT_S8_UINT, vk.FORMAT_D24_UNORM_S8_UINT, vk.FORMAT_D16_UNORM_S8_UINT)

    depth_format = None
    for valid_depth_format in valid_depth_formats:
        format_properties = vk.FormatProperties()
        tetris.GetPhysicalDeviceFormatProperties(physical_device, valid_depth_format, byref(format_properties))

        if IntFlag(vk.FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) in IntFlag(format_properties.optimal_tiling_features):
            depth_format = valid_depth_format
            break

    if depth_format is None:
        raise RuntimeError("Impossible to find a valid depth format")

    # Image
    image_info = vk.ImageCreateInfo(
        type = vk.STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        next = None,
        flags = 0,
        image_type = vk.IMAGE_TYPE_2D,
        format = depth_format,
        extent = vk.Extent3D(width=width, height=height, depth=1),
        mip_levels = 1,
        array_layers = 1,
        samples = vk.SAMPLE_COUNT_1_BIT,
        tiling = vk.IMAGE_TILING_OPTIMAL,
        usage = vk.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        sharing_mode = vk.SHARING_MODE_EXCLUSIVE,
        queue_family_index_count = 0,
        queue_family_indices = None,
        initial_layout = vk.IMAGE_LAYOUT_UNDEFINED
    )

    depth_image = call_vk(tetris.CreateImage, vk.Image, image_info, parent=device)
    
    # Image memory
    req = vk.MemoryRequirements()
    tetris.GetImageMemoryRequirements(device, depth_image, byref(req))

    mem_info = vk.MemoryAllocateInfo(
		type = vk.STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		next = None,
		allocation_size = req.size,
		memory_type_index = memory_type_index(tetris, vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
	)

    depth_memory = call_vk(tetris.AllocateMemory, vk.DeviceMemory, mem_info, parent=device)
    tetris.BindImageMemory(device, depth_image, depth_memory, 0)

    # View
    view_info = vk.ImageViewCreateInfo(
        type = vk.STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        next = None,
        flags = 0,
        image = depth_image,
        view_type = vk.IMAGE_VIEW_TYPE_2D,
        format = depth_format,
        components = vk.ComponentMapping(),
        subresource_range = vk.ImageSubresourceRange(
            aspect_mask = vk.IMAGE_ASPECT_DEPTH_BIT|vk.IMAGE_ASPECT_STENCIL_BIT,
            base_mip_level = 0,
            level_count = 1,
            base_array_layer = 0,
            layer_count = 1,
        )
    )
    depth_view = call_vk(tetris.CreateImageView, vk.ImageView, view_info, parent=device)

    if hasattr(tetris, 'depth_format'):
        tetris.DestroyImageView(device, tetris.depth_view, None)
        tetris.FreeMemory(device, tetris.depth_memory, None)
        tetris.DestroyImage(device, tetris.depth_image, None)
        
    tetris.depth_format = depth_format
    tetris.depth_image = depth_image
    tetris.depth_memory = depth_memory
    tetris.depth_view = depth_view


def create_render_pass(tetris):
    image_format = tetris.swapchain_format
    depth_format = tetris.depth_format
    
    # Attachments
    color = vk.AttachmentDescription(
        flags = 0,
        format = image_format,
        samples = vk.SAMPLE_COUNT_1_BIT,
        load_op = vk.ATTACHMENT_LOAD_OP_CLEAR,
        store_op = vk.ATTACHMENT_STORE_OP_STORE,
        stencil_load_op = vk.ATTACHMENT_LOAD_OP_DONT_CARE,
        stencil_store_op = vk.ATTACHMENT_STORE_OP_DONT_CARE,
        initial_layout = vk.IMAGE_LAYOUT_UNDEFINED,
        final_layout = vk.IMAGE_LAYOUT_PRESENT_SRC_KHR
    )

    depth = vk.AttachmentDescription(
        flags = 0,
        format = depth_format,
        samples = vk.SAMPLE_COUNT_1_BIT,
        load_op = vk.ATTACHMENT_LOAD_OP_CLEAR,
        store_op = vk.ATTACHMENT_STORE_OP_STORE,
        stencil_load_op = vk.ATTACHMENT_LOAD_OP_DONT_CARE,
        stencil_store_op = vk.ATTACHMENT_STORE_OP_DONT_CARE,
        initial_layout = vk.IMAGE_LAYOUT_UNDEFINED,
        final_layout = vk.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    )

    # Subpass
    color_ref = vk.AttachmentReference(attachment=0, layout=vk.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
    depth_ref = vk.AttachmentReference(attachment=1, layout=vk.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)

    colors_attachments = array(vk.AttachmentReference, 1, (color_ref,))

    subpass_info = vk.SubpassDescription(
        flags = 0,
        pipeline_bind_point = vk.PIPELINE_BIND_POINT_GRAPHICS,
        input_attachment_count = 0,
        input_attachments = None,
        color_attachment_count = 1,
        color_attachments = array_pointer(colors_attachments),
        resolve_attachments = None,
        depth_stencil_attachment = pointer(depth_ref),
        preserve_attachment_count = 0,
        preserve_attachments = None
    )

    # Renderpass dependencies
    prepare_drawing = vk.SubpassDependency(
        src_subpass = vk.SUBPASS_EXTERNAL,
        dst_subpass = 0,
        src_stage_mask = vk.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        dst_stage_mask = vk.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        src_access_mask = vk.ACCESS_MEMORY_READ_BIT,
        dst_access_mask = vk.ACCESS_COLOR_ATTACHMENT_READ_BIT | vk.ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        dependency_flags = vk.DEPENDENCY_BY_REGION_BIT
    )

    prepare_present = vk.SubpassDependency(
        src_subpass = 0,
        dst_subpass = vk.SUBPASS_EXTERNAL,
        src_stage_mask = vk.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        dst_stage_mask = vk.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        src_access_mask = vk.ACCESS_COLOR_ATTACHMENT_READ_BIT | vk.ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        dst_access_mask = vk.ACCESS_MEMORY_READ_BIT,
        dependency_flags = vk.DEPENDENCY_BY_REGION_BIT
    )

    # Render pass
    attachments = (color, depth)
    attachments_array = array(vk.AttachmentDescription, len(attachments), attachments)

    subpasses = (subpass_info,)
    subpasses_array = array(vk.SubpassDescription, len(subpasses), subpasses)

    dep = (prepare_drawing, prepare_present)
    dep_array = array(vk.SubpassDependency, len(dep), dep)

    render_pass_info = vk.RenderPassCreateInfo(
        type = vk.STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        next = None,
        flags = 0,
        attachment_count = len(attachments),
        attachments = array_pointer(attachments_array),
        subpass_count = len(subpasses),
        subpasses = array_pointer(subpasses_array),
        dependency_count = len(dep),
        dependencies = array_pointer(dep_array)
    )

    tetris.render_pass = call_vk(tetris.CreateRenderPass, vk.RenderPass, render_pass_info, tetris.device)


def create_framebuffers(tetris):
    device, extent = tetris.device, tetris.extent
    width, height = extent.width, extent.height
    depth_view = tetris.depth_view
    attachments = array(vk.ImageView, 2, ())

    fb_info = vk.FramebufferCreateInfo(
        type = vk.STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        next = None, 
        flags = 0,
        render_pass = tetris.render_pass,
        attachment_count = 2,
        attachments = array_pointer(attachments),
        width = width,
        height = height,
        layers = 1
    )

    framebuffers = []
    for _, view in tetris.swapchain_images:
        attachments[::] = (view, depth_view)
        framebuffers.append(call_vk(tetris.CreateFramebuffer, vk.Framebuffer, fb_info, device))
    
    if hasattr(tetris, 'framebuffers'):
        for fb in tetris.framebuffers:
            tetris.DestroyFramebuffer(device, fb, None)

    tetris.framebuffers = tuple(framebuffers)

#
# Vulkan base resources setup.
#

def create_semaphores(tetris):
    device = tetris.device
    sm_info = vk.SemaphoreCreateInfo(
        type = vk.STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        next = None,
        flags = 0
    )

    image_ready = call_vk(tetris.CreateSemaphore, vk.Semaphore, sm_info, device)
    rendering_done = call_vk(tetris.CreateSemaphore, vk.Semaphore, sm_info, device)

    tetris.semaphores = (image_ready, rendering_done)


def create_fences(tetris):
    device = tetris.device
    fence_info = vk.FenceCreateInfo(
        type = vk.STRUCTURE_TYPE_FENCE_CREATE_INFO,
        next = None,
        flags = vk.FENCE_CREATE_SIGNALED_BIT
    )

    fences = []
    for _ in tetris.framebuffers:
        fences.append(call_vk(tetris.CreateFence, vk.Fence, fence_info, device))

    tetris.fences = tuple(fences)


def create_commands(tetris):
    device = tetris.device
    fb_count = len(tetris.framebuffers)

    # Command pool
    pool_info = vk.CommandPoolCreateInfo(
        type = vk.STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        next = None,
        flags = vk.COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        queue_family_index = tetris.graphics.family_index
    )

    pool = call_vk(tetris.CreateCommandPool, vk.CommandPool, pool_info, device)

    # Command buffers
    # One for each framebuffer + 1 for setup purpose
    allocate_info = vk.CommandBufferAllocateInfo(
        type = vk.STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        next = None,
        command_pool = pool,
        level = vk.COMMAND_BUFFER_LEVEL_PRIMARY,
        command_buffer_count = fb_count + 1
    )

    commands = array(vk.CommandBuffer, fb_count + 1, ())
    tetris.AllocateCommandBuffers(device, allocate_info, commands)

    tetris.command_pool = pool
    tetris.commands = tuple( (i, vk.CommandBuffer(b)) for i, b in enumerate(commands[:fb_count]))
    tetris.setup_command = vk.CommandBuffer(commands[-1])


#
# Vulkan buffers setup
#

def create_staging_buffer(tetris, allocation_size):
    device = tetris.device

    # Buffer creation
    staging_buffer = create_buffer(tetris, buffer_params(
        allocation_size,
        vk.BUFFER_USAGE_TRANSFER_SRC_BIT
    ))

    staging_req = vk.MemoryRequirements()
    tetris.GetBufferMemoryRequirements(device, staging_buffer, byref(staging_req))

    staging_info = vk.MemoryAllocateInfo(
		type = vk.STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		next = None,
		allocation_size = staging_req.size,
		memory_type_index = memory_type_index(tetris, vk.MEMORY_PROPERTY_HOST_COHERENT_BIT | vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT)
	)

    staging_alloc = call_vk(tetris.AllocateMemory, vk.DeviceMemory, staging_info, device)

    tetris.BindBufferMemory(device, staging_buffer, staging_alloc, 0)

    # Make sure the data is zeroed before uploading it to the GPU
    ptr = c_void_p(0)
    result = tetris.MapMemory(device, staging_alloc, 0, allocation_size, 0, byref(ptr))
    if result != vk.SUCCESS:
        raise RuntimeError("Failed to map memory")

    memset(ptr, 0, allocation_size)

    tetris.UnmapMemory(device, staging_alloc)

    return staging_buffer, staging_alloc


def upload_staging(tetris, staging_buffer, buffer_regions):
    cmd = tetris.setup_command
    begin_info = vk.CommandBufferBeginInfo(
        type = vk.STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        next = None,
        flags = 0,
        inheritance_info = None
    )

    tetris.BeginCommandBuffer(cmd, begin_info)

    for buffer, region in buffer_regions:
        tetris.CmdCopyBuffer(cmd, staging_buffer, buffer, 1, byref(region))

    tetris.EndCommandBuffer(cmd)

    submit = vk.SubmitInfo(
        type = vk.STRUCTURE_TYPE_SUBMIT_INFO,
        next = None,
        wait_semaphore_count = 0,
        wait_semaphores = None,
        wait_dst_stage_mask = None,
        command_buffer_count = 1,
        command_buffers = pointer(cmd),
        signal_semaphore_count = 0,
        signal_semaphores = None
    )

    tetris.QueueSubmit(tetris.graphics.handle, 1, byref(submit), 0)
    tetris.QueueWaitIdle(tetris.graphics.handle)


def create_buffers(tetris):
    device = tetris.device
    attr_type = array(c_float, 4)
    DRAW_STRUCT_SIZE = sizeof(vk.DrawIndexedIndirectCommand)

    # VERY VERY IMPORTANT. SSBO aligment must be respected. I keep forgeting this and all my demo crash on NVdia hardware.
    storage_align = tetris.limits.min_storage_buffer_offset_alignment

    # Compute allocation space
    draw_count_size = align(4, storage_align)                                           # Starts at 4 bytes for the count buffer
    draw_info_size  = align(DRAW_STRUCT_SIZE * MAX_OBJECT_COUNT, storage_align)         # Allocate space for the indirect draw commands
    attributes_size = align(sizeof(attr_type) * MAX_ATTRIBUTES_COUNT, storage_align)    # Enough to hold 1000 vertices. Should be plenty
    indices_size    = align(sizeof(c_uint16) * MAX_INDICES_COUNT, storage_align)
    game_data_size  = align(sizeof(GameData), storage_align)

    # Device setup
    draw_info_buffer = create_buffer(tetris, buffer_params (
        draw_count_size + draw_info_size,
        vk.BUFFER_USAGE_INDIRECT_BUFFER_BIT | vk.BUFFER_USAGE_TRANSFER_DST_BIT | vk.BUFFER_USAGE_STORAGE_BUFFER_BIT
    ))

    attributes_buffer = create_buffer(tetris, buffer_params (
        attributes_size + indices_size,
        vk.BUFFER_USAGE_VERTEX_BUFFER_BIT | vk.BUFFER_USAGE_INDEX_BUFFER_BIT | vk.BUFFER_USAGE_TRANSFER_DST_BIT | vk.BUFFER_USAGE_STORAGE_BUFFER_BIT
    ))

    game_data_buffer = create_buffer(tetris, buffer_params(
        game_data_size,
        vk.BUFFER_USAGE_TRANSFER_DST_BIT | vk.BUFFER_USAGE_STORAGE_BUFFER_BIT
    ))

    # Offsets
    index_offset = 0
    positions_offset = align(indices_size, storage_align)

    draw_count_offset = 0
    draw_params_offset = draw_count_size

    draw_data = {
        "index_offset": index_offset,
        "index_size": indices_size,

        "positions_offset": positions_offset,
        "positions_size": attributes_size,

        "vertex_buffers": array(vk.Buffer, 1, (attributes_buffer,)),
        "vertex_offsets": array(vk.DeviceSize, 1, (positions_offset,)),

        "draw_params_offset": draw_count_size,
        "draw_count_offset": 0
    }

    # Final device memory allocation
    req = vk.MemoryRequirements()
    buffers, offsets = (draw_info_buffer, attributes_buffer, game_data_buffer), [0,0,0]
    buffer_copy_regions = []
    current_offset = staging_alloc_size = 0

    for index, buffer in enumerate(buffers):
        tetris.GetBufferMemoryRequirements(device, buffer, byref(req))

        aligned_offset = align(current_offset, max(req.alignment, storage_align))
        offsets[index] = aligned_offset

        buffer_copy_regions.append((buffer, vk.BufferCopy(src_offset=0, dst_offset=0, size=req.size)))
        staging_alloc_size = max(staging_alloc_size, req.size)

        current_offset = align(aligned_offset + req.size, max(req.alignment, storage_align))
    
    final_info = vk.MemoryAllocateInfo(
		type = vk.STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		next = None,
		allocation_size = current_offset + req.size,
		memory_type_index = memory_type_index(tetris, vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
	)

    final_alloc = call_vk(tetris.AllocateMemory, vk.DeviceMemory, final_info, device)

    for buffer, offset in zip(buffers, offsets):
        tetris.BindBufferMemory(device, buffer, final_alloc, offset)

    # Mem set device memory to 0
    staging_buffer, staging_alloc = create_staging_buffer(tetris, staging_alloc_size)
    upload_staging(tetris, staging_buffer, buffer_copy_regions)

    # Printing out data informations
    print("")
    print(f"Staging alloc size: {staging_alloc_size}")
    print(f"Final alloc size: {final_info.allocation_size}")
    print(f"Draw info buffer size: {draw_count_size + draw_info_size}")
    print(f"Attributes buffer size: {attributes_size + indices_size}")
    print(f"Game data buffer size: {game_data_size}")

    # Saving
    tetris.draw_info_buffer = draw_info_buffer
    tetris.attributes_buffer = attributes_buffer
    tetris.game_data_buffer = game_data_buffer
    
    tetris.final_alloc = final_alloc
    tetris.draw_data = draw_data

    tetris.max_obj_count = MAX_OBJECT_COUNT
    tetris.max_indices_count = MAX_INDICES_COUNT
    tetris.max_attributes_count = MAX_ATTRIBUTES_COUNT

    # Staging cleanup
    tetris.DestroyBuffer(device, staging_buffer, None)
    tetris.FreeMemory(device, staging_alloc, None)


def create_game_state_buffer(tetris):
    # Note that this is not part of "create_buffers" because the method started to make my head hurt.
    device = tetris.device

    # Buffer
    game_state_buffer = create_buffer(tetris, buffer_params(
        sizeof(GameState),
        vk.BUFFER_USAGE_TRANSFER_DST_BIT | vk.BUFFER_USAGE_STORAGE_BUFFER_BIT
    ))
    
    # Memory
    req = vk.MemoryRequirements()
    tetris.GetBufferMemoryRequirements(device, game_state_buffer, byref(req))   

    final_info = vk.MemoryAllocateInfo(
		type = vk.STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		next = None,
		allocation_size = req.size,
		memory_type_index = memory_type_index(tetris, vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT)
	)

    game_state_alloc = call_vk(tetris.AllocateMemory, vk.DeviceMemory, final_info, device)

    # Memory binding
    tetris.BindBufferMemory(device, game_state_buffer, game_state_alloc, 0)

    # Zeroing
    ptr = c_void_p(0)
    result = tetris.MapMemory(device, game_state_alloc, 0, req.size, 0, byref(ptr))
    if result != vk.SUCCESS:
        raise RuntimeError("Failed to map game state memory")

    memset(ptr, 0, req.size)

    tetris.UnmapMemory(device, game_state_alloc)

    # Printing
    print(f"Game state buffer size: {sizeof(GameState)}")
    print("")

    # Saving
    tetris.game_state_buffer = game_state_buffer
    tetris.game_state_alloc = game_state_alloc
    tetris.GameState = GameState

#
# Vulkan pipeline & shaders setup
#

def create_shaders(tetris):
    device = tetris.device
    create_shader = lambda info: call_vk(tetris.CreateShaderModule, vk.ShaderModule, info, parent=device)

    codes = (
        open('./tetris.vert.spv', 'rb').read(),
        open('./tetris.frag.spv', 'rb').read(),
        open('./tetris.comp.spv', 'rb').read(),
        open('./tetris_init.comp.spv', 'rb').read(),
    )

    shader_infos = []
    for code in codes:
        shader_infos.append(vk.ShaderModuleCreateInfo(
            type = vk.STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            next = None,
            flags = 0,
            code_size = len(code),
            code = array_pointer(code, c_uint32)
        ))

    offset = sizeof(vk.ShaderModuleCreateInfo)
    vert_module = create_shader(shader_infos[0])
    frag_module = create_shader(shader_infos[1])
    comp_module = create_shader(shader_infos[2])
    comp_init_module = create_shader(shader_infos[3])

    tetris.shader_compute = comp_module
    tetris.shader_compute_init = comp_init_module
    tetris.shader_render = (
        (vert_module, vk.SHADER_STAGE_VERTEX_BIT),
        (frag_module, vk.SHADER_STAGE_FRAGMENT_BIT)
    )


def create_descriptor_set_layouts(tetris):
    def layout_binding(binding, stage=vk.SHADER_STAGE_COMPUTE_BIT):
        return vk.DescriptorSetLayoutBinding(
            binding = binding,
            descriptor_type = vk.DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count = 1,
            stage_flags = stage,
            immutable_samplers = None
        )

    # Compute set layouts
    bindings = array(vk.DescriptorSetLayoutBinding, 6, (
        layout_binding(0),   # Draw commands
        layout_binding(1),   # Draw count
        layout_binding(2),   # Index
        layout_binding(3),   # Attributes
        layout_binding(4),   # Game data
        layout_binding(5),   # Game State
    ))

    set_layout_info = vk.DescriptorSetLayoutCreateInfo(
        type = vk.STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        next = None,
        flags = 0,
        binding_count = len(bindings),
        bindings = array_pointer(bindings)
    )

    tetris.compute_set_layouts = (
        call_vk(tetris.CreateDescriptorSetLayout, vk.DescriptorSetLayout, set_layout_info, tetris.device),
    )

    # Render set layouts
    bindings = array(vk.DescriptorSetLayoutBinding, 1, (
        layout_binding(0, vk.SHADER_STAGE_COMPUTE_BIT | vk.SHADER_STAGE_VERTEX_BIT),      # Game data
    ))

    set_layout_info.binding_count = len(bindings)
    set_layout_info.bindings = array_pointer(bindings)

    tetris.render_set_layouts = (
        call_vk(tetris.CreateDescriptorSetLayout, vk.DescriptorSetLayout, set_layout_info, tetris.device),
    )


def create_descriptors(tetris):
    device = tetris.device
    MAX_SETS = len(tetris.compute_set_layouts) + len(tetris.render_set_layouts)

    # Descriptor pool
    pool_sizes = array(vk.DescriptorPoolSize, 1, (
        vk.DescriptorPoolSize(type=vk.DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptor_count=7),
    ))

    descriptor_pool_info = vk.DescriptorPoolCreateInfo(
        type = vk.STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        next = None,
        flags = 0,
        max_sets = MAX_SETS,
        pool_size_count = len(pool_sizes),
        pool_sizes = array_pointer(pool_sizes)
    )

    descriptor_pool = call_vk(tetris.CreateDescriptorPool, vk.DescriptorPool, descriptor_pool_info, device)

    sets = tetris.render_set_layouts + tetris.compute_set_layouts
    sets_layouts_array = array(vk.DescriptorSetLayout, MAX_SETS, sets)
    allocate_info = vk.DescriptorSetAllocateInfo(
        type = vk.STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        next = None,
        descriptor_pool = descriptor_pool,
        descriptor_set_count = MAX_SETS,
        set_layouts = array_pointer(sets_layouts_array)
    )

    descriptor_sets = array(vk.DescriptorSet, MAX_SETS, ())
    tetris.AllocateDescriptorSets(device, allocate_info, array_pointer(descriptor_sets))

    tetris.descriptor_pool = descriptor_pool
    tetris.render_descriptor_sets = array(vk.DescriptorSet, 1, descriptor_sets[0:1])
    tetris.compute_descriptor_sets =array(vk.DescriptorSet, 1,  descriptor_sets[1:2])


def update_descriptor_sets(tetris):
    def buffer_write_set(dst_set, dst_binding, buffer):
        return vk.WriteDescriptorSet(
            type = vk.STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            next = None,
            dst_set = dst_set,
            dst_binding = dst_binding,
            dst_array_element = 0,
            descriptor_count = 1,
            descriptor_type = vk.DESCRIPTOR_TYPE_STORAGE_BUFFER,
            image_info = None,
            buffer_info = pointer(buffer),
            texel_buffer_view = None
        )

    dd = tetris.draw_data
    draw_info_buffer = tetris.draw_info_buffer
    attributes_buffer = tetris.attributes_buffer
    game_data_buffer = tetris.game_data_buffer
    game_state_buffer = tetris.game_state_buffer

    compute_set = tetris.compute_descriptor_sets[0]
    render_set = tetris.render_descriptor_sets[0]

    # Graphics set
    draw_commands_binding = buffer_write_set(
        compute_set, 0,
        vk.DescriptorBufferInfo(buffer=draw_info_buffer, offset=dd["draw_params_offset"], range=vk.WHOLE_SIZE)
    )

    draw_commands_count_binding = buffer_write_set(
        compute_set, 1,
        vk.DescriptorBufferInfo(buffer=draw_info_buffer, offset=dd["draw_count_offset"], range=sizeof(c_uint32))
    )

    indices_binding = buffer_write_set(
        compute_set, 2,
        vk.DescriptorBufferInfo(buffer=attributes_buffer, offset=dd["index_offset"], range=dd["index_size"])
    )

    attributes_binding = buffer_write_set(
        compute_set, 3,
        vk.DescriptorBufferInfo(buffer=attributes_buffer, offset=dd["positions_offset"], range=dd["positions_size"])
    )

    game_data_binding = buffer_write_set(
        compute_set, 4,
        vk.DescriptorBufferInfo(buffer=game_data_buffer, offset=0, range=vk.WHOLE_SIZE)
    )

    game_state_binding = buffer_write_set(
        compute_set, 5,
        vk.DescriptorBufferInfo(buffer=game_state_buffer, offset=0, range=vk.WHOLE_SIZE)
    )

    # Render Set
    game_data_binding_vertex = buffer_write_set(
        render_set, 0,
        vk.DescriptorBufferInfo(buffer=game_data_buffer, offset=0, range=vk.WHOLE_SIZE)
    )

    writes = [
        draw_commands_binding, draw_commands_count_binding, indices_binding, attributes_binding, game_data_binding,
        game_data_binding_vertex, game_state_binding
    ]

    writes_array = array(vk.WriteDescriptorSet, len(writes), writes)
    tetris.UpdateDescriptorSets(tetris.device, len(writes), array_pointer(writes_array), 0, None)


def create_pipeline_layouts(tetris):
    device = tetris.device
    
    render_set_layouts_array = array(vk.DescriptorSetLayout, len(tetris.render_set_layouts), tetris.render_set_layouts)
    compute_set_layouts_array = array(vk.DescriptorSetLayout, len(tetris.compute_set_layouts), tetris.compute_set_layouts)

    render_layout_info = vk.PipelineLayoutCreateInfo(
        type = vk.STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        next = None,
        flags = 0,
        set_layout_count = len(render_set_layouts_array),
        set_layouts = array_pointer(render_set_layouts_array),
        push_constant_range_count = 0,
        push_constant_ranges = None
    )

    compute_layout_info = vk.PipelineLayoutCreateInfo(
        type = vk.STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        next = None,
        flags = 0,
        set_layout_count = len(compute_set_layouts_array),
        set_layouts = array_pointer(compute_set_layouts_array),
        push_constant_range_count = 0,
        push_constant_ranges = None
    )

    tetris.render_pipeline_layout = call_vk(tetris.CreatePipelineLayout, vk.PipelineLayout, render_layout_info, device)
    tetris.compute_pipeline_layout = call_vk(tetris.CreatePipelineLayout, vk.PipelineLayout, compute_layout_info, device)

    
def create_compute_pipeline(tetris):
    entry_name = bytes('main', 'utf-8') + b'\x00'
    int_size = sizeof(c_uint32)

    max_obj_count = vk.SpecializationMapEntry(constant_ID=1, offset=0, size=int_size)
    max_indices = vk.SpecializationMapEntry(constant_ID=2, offset=int_size, size=int_size)
    max_attributes = vk.SpecializationMapEntry(constant_ID=3, offset=int_size*2, size=int_size)
    map_entries = array(vk.SpecializationMapEntry, 3, (max_obj_count, max_indices, max_attributes))

    # Constant values are defined in `create_buffers`
    constant_buffer = array(c_uint32, 3, (tetris.max_obj_count, tetris.max_indices_count, tetris.max_attributes_count))
    constant_buffer_ptr = cast(array_pointer(constant_buffer), c_void_p)

    spez = vk.SpecializationInfo(
        map_entry_count = len(map_entries),
        map_entries = array_pointer(map_entries),
        data_size = len(constant_buffer) * sizeof(c_uint32),
        data = constant_buffer_ptr
    )

    compute_stage = vk.PipelineShaderStageCreateInfo(
        type = vk.STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        next = None,
        flags = 0,
        stage = vk.SHADER_STAGE_COMPUTE_BIT,
        module = tetris.shader_compute,
        name = c_char_p(entry_name),
        specialization_info = pointer(spez)
    )
    
    compute_pipeline_info = vk.ComputePipelineCreateInfo(
        type = vk.STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        next = None,
        flags = 0,
        stage = compute_stage,
        layout = tetris.compute_pipeline_layout,
        base_pipeline_handle = 0,
        base_pipeline_index = -1
    )

    tetris.compute_pipeline = vk.Pipeline(0)
    result = tetris.CreateComputePipelines(tetris.device, 0, 1, byref(compute_pipeline_info), None, byref(tetris.compute_pipeline))
    if result != vk.SUCCESS:
        raise RuntimeError(f"Failed to create the compute pipeline: 0x{result:X}")

    compute_pipeline_info.stage.module = tetris.shader_compute_init
    tetris.compute_init_pipeline = vk.Pipeline(0)
    result = tetris.CreateComputePipelines(tetris.device, 0, 1, byref(compute_pipeline_info), None, byref(tetris.compute_init_pipeline))
    if result != vk.SUCCESS:
        raise RuntimeError(f"Failed to create the compute init pipeline: 0x{result:X}")


def create_render_pipeline(tetris):
    device = tetris.device

    # Shader states
    entry_name = bytes('main', 'utf-8') + b'\x00'
    shader_stages = []

    for module, stage_flags in tetris.shader_render:
        stage = vk.PipelineShaderStageCreateInfo(
            type = vk.STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            next = None,
            flags = 0,
            stage = stage_flags,
            module = module,
            name = c_char_p(entry_name),
            specialization_info = None
        )

        shader_stages.append(stage)

    shader_stages_array = array(vk.PipelineShaderStageCreateInfo, len(shader_stages), shader_stages)

    # Vertex Input state
    bindings = (
        vk.VertexInputBindingDescription( binding=0, stride=sizeof(c_float)*4, input_rate=vk.VERTEX_INPUT_RATE_VERTEX),
    )

    attributes = (
        vk.VertexInputAttributeDescription(location = 0, binding = 0, format = vk.FORMAT_R32G32B32A32_SFLOAT, offset = 0),   # Position
    )

    bindings_array = array(vk.VertexInputBindingDescription, len(bindings), bindings)
    attributes_array = array(vk.VertexInputAttributeDescription, len(attributes), attributes)
    vertex_input_state = vk.PipelineVertexInputStateCreateInfo(
        type = vk.STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        next = None,
        flags = 0,
        vertex_binding_description_count = len(bindings),
        vertex_binding_descriptions = array_pointer(bindings_array),
        vertex_attribute_description_count = len(attributes),
        vertex_attribute_descriptions = array_pointer(attributes_array)
    )

    # Other pipeline states
    depth_stencil_state = vk.PipelineDepthStencilStateCreateInfo(
        type = vk.STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        next = None,
        flags = 0,
        depth_test_enable = vk.TRUE,
        depth_write_enable = vk.TRUE,
        depth_compare_op = vk.COMPARE_OP_LESS_OR_EQUAL, 
        depth_bounds_test_enable = vk.FALSE, 
        stencil_test_enable = vk.FALSE, 
        front = vk.StencilOpState(
            fail_op = vk.STENCIL_OP_KEEP,
            pass_op = vk.STENCIL_OP_KEEP,
            depth_fail_op = vk.STENCIL_OP_KEEP,
            compare_op = vk.COMPARE_OP_ALWAYS,
            compare_mask = vk.STENCIL_OP_KEEP,
            write_mask =vk.STENCIL_OP_KEEP,
            reference = vk.STENCIL_OP_KEEP
        ), 
        back = vk.StencilOpState(
            fail_op = vk.STENCIL_OP_KEEP,
            pass_op = vk.STENCIL_OP_KEEP,
            depth_fail_op = vk.STENCIL_OP_KEEP,
            compare_op = vk.COMPARE_OP_ALWAYS,
            compare_mask = vk.STENCIL_OP_KEEP,
            write_mask =vk.STENCIL_OP_KEEP,
            reference = vk.STENCIL_OP_KEEP
        ),
        min_depth_bounds = 0.0, 
        max_depth_bounds = 1.0
    )

    input_assembly_state = vk.PipelineInputAssemblyStateCreateInfo(
        type = vk.STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        next = None,
        flags = 0,
        topology = vk.PRIMITIVE_TOPOLOGY_LINE_STRIP,
        primitive_restart_enable = vk.FALSE
    )

    multisample_state = vk.PipelineMultisampleStateCreateInfo(
        type = vk.STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        next = None,
        flags = 0,
        rasterization_samples = vk.SAMPLE_COUNT_1_BIT,
        sample_shading_enable = vk.FALSE,
        min_sample_shading = 0.0,
        sample_mask = None,
        alpha_toCoverage_enable = vk.FALSE,
        alpha_toOne_enable = vk.FALSE
    )

    # Viewports and scissors are dynamic
    viewport_state = vk.PipelineViewportStateCreateInfo(
        type = vk.STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        next = None,
        flags = 0,
        viewport_count = 1,
        viewports = None,
        scissor_count = 1,
        scissors = None
    )

    rasterization_state = vk.PipelineRasterizationStateCreateInfo(
        type = vk.STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        next = None,
        flags = 0,
        depth_clamp_enable = vk.FALSE,
        rasterizer_discard_enable = vk.FALSE,
        polygon_mode = vk.POLYGON_MODE_FILL,
        cull_mode = vk.CULL_MODE_NONE,
        front_face = vk.FRONT_FACE_CLOCKWISE,
        depth_bias_enable = vk.FALSE,
        depth_bias_constant_factor = 0,
        depth_bias_clamp = 0.0,
        depth_bias_slope_factor = 0.0,
        line_width = 1.0,
    )

    blend_attachment = vk.PipelineColorBlendAttachmentState(
        blend_enable = vk.FALSE,
        src_color_blend_factor = 0,
        dst_color_blend_factor = 0,
        color_blend_op = 0,
        src_alpha_blend_factor = 0,
        dst_alpha_blend_factor = 0,
        alpha_blend_op = 0,
        color_write_mask = 0xF
    )

    blend_attachments = array(vk.PipelineColorBlendAttachmentState, 1, (blend_attachment,))
    color_blend_state = vk.PipelineColorBlendStateCreateInfo(
        type = vk.STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        next = None,
        flags = 0,
        logic_opEnable =False,
        logic_op = 0,
        attachment_count = 1,
        attachments = array_pointer(blend_attachments),
        blend_constants = array(c_float, 4, (0,0,0,0))
    )

    states_array = array(c_uint32, 2, (vk.DYNAMIC_STATE_VIEWPORT, vk.DYNAMIC_STATE_SCISSOR))
    dynamic_state = vk.PipelineDynamicStateCreateInfo(
        type = vk.STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        next = None,
        flags = 0,
        dynamic_state_count = 2,
        dynamic_states = array_pointer(states_array)
    )

    # Pipeline creation
    pipeline_info = vk.GraphicsPipelineCreateInfo(
        type = vk.STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        next = None,
        flags = 0,
        stage_count = len(shader_stages),
        stages = array_pointer(shader_stages_array),
        vertex_input_state = pointer(vertex_input_state),
        input_assembly_state = pointer(input_assembly_state),
        tessellation_state = None,
        viewport_state = pointer(viewport_state),
        rasterization_state = pointer(rasterization_state),
        multisample_state = pointer(multisample_state),
        depth_stencil_state = pointer(depth_stencil_state),
        color_blend_state = pointer(color_blend_state),
        dynamic_state = pointer(dynamic_state),
        layout = tetris.render_pipeline_layout,
        render_pass = tetris.render_pass,
        subpass = 0,
        base_pipeline_handle = 0,
        base_pipeline_index = 0
    )

    if hasattr(tetris, 'pipeline'):
        tetris.DestroyPipeline(device, tetris.pipeline, None)

    tetris.pipeline = vk.Pipeline(0)
    result = tetris.CreateGraphicsPipelines(device, 0, 1, byref(pipeline_info), None, byref(tetris.pipeline))


def create_render_cache(tetris):
    # Cache ctypes structures to keep the run loop super lean
    extent = tetris.extent
    width, height = extent.width, extent.height
    render_pass = tetris.render_pass

    # Command buffer begin info. Technically not required to be cached because the command buffers
    # are only recorded once. But I do cache it in my other projects.
    begin_info = vk.CommandBufferBeginInfo(
        type = vk.STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        next = None,
        flags = 0,
        inheritance_info = None
    )

    # Clear values
    black = array(c_float, 4, (0.0, 0.0, 0.0, 0.0))
    clear_values = array(vk.ClearValue, 2, ())
    clear_values[0] = vk.ClearValue(color = vk.ClearColorValue(float32=black))
    clear_values[1] = vk.ClearValue(depth_stencil = vk.ClearDepthStencilValue(depth=1.0, stencil=0))

    # Scissors & Viewports
    scissor = vk.Rect2D(offset=vk.Offset2D(x=0, y=0), extent=vk.Extent2D(width=width, height=height))
    viewport = vk.Viewport(x=0, y=0, width=width, height=height, min_depth=0.0, max_depth=1.0)

    # Render pass begin info
    # Same deal as the command buffer begin info
    render_pass_begins = []
    for fb in tetris.framebuffers:
        render_pass_begins.append(vk.RenderPassBeginInfo(
            type = vk.STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            next = None,
            render_pass = render_pass,
            framebuffer = fb,
            render_area = vk.Rect2D(
                offset=vk.Offset2D(x=0, y=0),
                extent=vk.Extent2D(width=extent.width, height=extent.height)
            ),
            clear_value_count = 2,
            clear_values = array_pointer(clear_values)
        ))

    # Submit Infos
    # This one is actually worthwhile to cache in this application
    image_ready_sm_ptr = pointer(tetris.semaphores[0])
    rendering_done_sm_ptr = pointer(tetris.semaphores[1])
    wait_dst = c_uint32(vk.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)

    submit_infos = []
    for _, cmd in tetris.commands:
        submit_infos.append(vk.SubmitInfo(
            type = vk.STRUCTURE_TYPE_SUBMIT_INFO,
            next = None,
            wait_semaphore_count = 1,
            wait_semaphores = image_ready_sm_ptr,
            wait_dst_stage_mask = pointer(wait_dst),
            command_buffer_count = 1,
            command_buffers = pointer(cmd),
            signal_semaphore_count = 1,
            signal_semaphores = rendering_done_sm_ptr
        ))

    # Present Infos
    # This one too is actually worthwhile to cache
    present_infos = []
    for i in range(len(tetris.framebuffers)):
        present_infos.append(vk.PresentInfoKHR(
            type = vk.STRUCTURE_TYPE_PRESENT_INFO_KHR,
            next = None,
            wait_semaphore_count = 1,
            wait_semaphores = rendering_done_sm_ptr,
            swapchain_count = 1,
            swapchains = pointer(tetris.swapchain),
            image_indices = pointer(c_uint32(i)),
            results = None
        ))


    tetris.render_cache = {
        "begin_info": begin_info,
        "clear_values": clear_values,
        "render_pass_begins": tuple(render_pass_begins),
        "image_index": c_uint32(-1),
        "submit_infos": tuple(submit_infos),
        "present_infos": tuple(present_infos),
        "scissor": scissor,
        "viewport": viewport
    } 


def record_draw_commands(tetris, index, cmd):
    # Render commands recording for device without a dedicated compute queue
    cache = tetris.render_cache

    tetris.BeginCommandBuffer(cmd, cache["begin_info"])
    draw_data = tetris.draw_data

    # Compute phase
    tetris.CmdBindPipeline(cmd, vk.PIPELINE_BIND_POINT_COMPUTE, tetris.compute_pipeline)
    tetris.CmdBindDescriptorSets(
        cmd, vk.PIPELINE_BIND_POINT_COMPUTE, tetris.compute_pipeline_layout,
        0, len(tetris.compute_descriptor_sets), array_pointer(tetris.compute_descriptor_sets),
        0, None 
    )
    tetris.CmdDispatch(cmd, 1, 1, 1)

    # Barrier to make sure the compute shader execution finished before the render pass begins
    # Not even sure I'm using this right though...
    tetris.CmdPipelineBarrier(
        cmd,
        vk.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        vk.PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        0,
        0, None,  # memory barriers
        0, None,  # buffer memory barriers
        0, None   # image memory barriers
    )

    # Render phase
    tetris.CmdBeginRenderPass(cmd, cache["render_pass_begins"][index], vk.SUBPASS_CONTENTS_INLINE)

    tetris.CmdBindPipeline(cmd, vk.PIPELINE_BIND_POINT_GRAPHICS, tetris.pipeline)
    tetris.CmdSetScissor(cmd, 0, 1, byref(cache['scissor']))
    tetris.CmdSetViewport(cmd, 0, 1, byref(cache['viewport']))

    tetris.CmdBindDescriptorSets(
        cmd, vk.PIPELINE_BIND_POINT_GRAPHICS, tetris.render_pipeline_layout,
        0, len(tetris.render_descriptor_sets), array_pointer(tetris.render_descriptor_sets),
        0, None 
    )

    if hasattr(tetris, 'CmdDrawIndexedIndirectCountKHR'):
        draw = tetris.CmdDrawIndexedIndirectCountKHR
    elif hasattr(tetris, 'CmdDrawIndexedIndirectCountAMD'):
        draw = tetris.CmdDrawIndexedIndirectCountAMD
    else:
        raise RuntimeError("Waht?")

    tetris.CmdBindIndexBuffer(cmd, tetris.attributes_buffer, draw_data["index_offset"], vk.INDEX_TYPE_UINT32)
    tetris.CmdBindVertexBuffers(cmd, 0, len(draw_data["vertex_buffers"]), draw_data["vertex_buffers"], draw_data["vertex_offsets"])

    draw(
        cmd, 
        tetris.draw_info_buffer, draw_data["draw_params_offset"], 
        tetris.draw_info_buffer, draw_data["draw_count_offset"],
        tetris.max_obj_count,
        sizeof(vk.DrawIndexedIndirectCommand)
    )

    tetris.CmdEndRenderPass(cmd)

    tetris.EndCommandBuffer(cmd)


def record_commands(tetris):
    for i, cmd in tetris.commands:
        record_draw_commands(tetris, i, cmd)

#
# Runtime
#

def init(tetris):
    cmd = tetris.setup_command

    tetris.BeginCommandBuffer(cmd, tetris.render_cache["begin_info"])

    tetris.CmdBindPipeline(cmd, vk.PIPELINE_BIND_POINT_COMPUTE, tetris.compute_init_pipeline)
    tetris.CmdBindDescriptorSets(
        cmd, vk.PIPELINE_BIND_POINT_COMPUTE, tetris.compute_pipeline_layout,
        0, len(tetris.compute_descriptor_sets), array_pointer(tetris.compute_descriptor_sets),
        0, None 
    )

    tetris.CmdDispatch(cmd, 1, 1, 1)

    tetris.EndCommandBuffer(cmd)

    submit = vk.SubmitInfo(
        type = vk.STRUCTURE_TYPE_SUBMIT_INFO,
        next = None,
        wait_semaphore_count = 0,
        wait_semaphores = None,
        wait_dst_stage_mask = None,
        command_buffer_count = 1,
        command_buffers = pointer(cmd),
        signal_semaphore_count = 0,
        signal_semaphores = None
    )

    tetris.QueueSubmit(tetris.graphics.handle, 1, byref(submit), 0)
    tetris.QueueWaitIdle(tetris.graphics.handle)


def update(tetris, base_time, input_state, window):
    device = tetris.device
    alloc = tetris.game_state_alloc

    # When a key is held on linux, the system generates a KeyDown and a KeyUp event but Windows
    # only generates a KeyDown. This method makes this code work on both OS
    input_buffer = {}  

    ptr = c_void_p(0)
    tetris.MapMemory(device, alloc, 0, vk.WHOLE_SIZE, 0, byref(ptr))

    # Update game time
    data = tetris.GameState.from_address(ptr.value)
    data.game_time = time.monotonic() - base_time

    # Update game input
    for event, event_data in window.events:
        if not (event is events.KeyPress or event is events.KeyRelease):
            continue
        elif event_data.key not in input_state.keys():
            continue
        
        key = event_data.key
        pressed = event is events.KeyPress
        input_state[key] = input_buffer.get(key, pressed)
        input_buffer[key] = pressed
    
    keys = events.Keys
    data.up = input_state[keys.Up]
    data.right = input_state[keys.Right]
    data.left = input_state[keys.Left]
    data.space = input_state[keys.Space]
 
    tetris.UnmapMemory(device, alloc)


def render(tetris, base_time, input_state, window):
    UINT32_MAX = 4294967295
    device, swapchain = tetris.device, tetris.swapchain
    image_ready, rendering_done = tetris.semaphores
    
    cache = tetris.render_cache
    image_index = cache['image_index']

    # Wait for presentable image
    if image_index.value == UINT32_MAX:
        result = tetris.AcquireNextImageKHR(device, swapchain, 0, image_ready, 0, byref(image_index))
        if result == vk.NOT_READY or result == vk.TIMEOUT:
            return  # Swapchain not ready. Skip rendering for this cycle
        elif result == vk.SUBOPTIMAL_KHR:
            raise NotImplementedError("SUBOPTIMAL_KHR. WONTFIX. should trigger swapchain rebuild, but too lazy to implement.")
        elif result != vk.SUCCESS:
            raise RuntimeError(f"AcquireNextImageKHR failed with error code 0x{result:X}")

    # Wait for fence
    image_index = image_index.value
    fence = tetris.fences[image_index]
    result = tetris.WaitForFences(device, 1, byref(fence), vk.TRUE, 0)
    
    if result == vk.TIMEOUT:
        return  # Fence not ready. Skip rendering for this cycle
    elif result != vk.SUCCESS:
        raise RuntimeError(f"WaitForFences failed with error code 0x{result:X}")
    else:
        tetris.ResetFences(device, 1, byref(fence))

    # Update inputs before submit
    update(tetris, base_time, input_state, window)

    # Submit
    queue = tetris.graphics.handle
    submit_info = cache['submit_infos'][image_index]
    result = tetris.QueueSubmit(queue, 1, byref(submit_info), fence)
    if result != vk.SUCCESS:
        raise RuntimeError(f"QueueSubmit failed with error code 0x{result:X}")

    # Present
    present_info = cache['present_infos'][image_index]
    result = tetris.QueuePresentKHR(queue, byref(present_info))
    if result != vk.SUCCESS:
        raise RuntimeError(f"QueuePresentKHR failed with error code 0x{result:X}")

    # Clear the image to try to acquire a presentable image for the next cycle
    cache['image_index'].value = UINT32_MAX


def run_loop(tetris):
    running = True
    w = tetris.window
    w.show()

    base_time = time.monotonic()

    input_state = {k: False for k in events.ArrowKeys}
    input_state[events.Keys.Space] = False

    update(tetris, base_time, input_state, w)
    init(tetris)

    print("It's running!")

    while running:
        w.translate_system_events()
        running = not w.must_exit 
        render(tetris, base_time, input_state, w)


def cleanup(tetris):
    i, d = tetris.instance, tetris.device

    tetris.DeviceWaitIdle(d)

    tetris.DestroyPipeline(d, tetris.pipeline, None)
    tetris.DestroyPipeline(d, tetris.compute_pipeline, None)
    tetris.DestroyPipeline(d, tetris.compute_init_pipeline, None)
    tetris.DestroyPipelineLayout(d, tetris.render_pipeline_layout, None)
    tetris.DestroyPipelineLayout(d, tetris.compute_pipeline_layout, None)

    tetris.DestroyDescriptorPool(d, tetris.descriptor_pool, None)

    tetris.DestroyDescriptorSetLayout(d, tetris.render_set_layouts[0], None)
    tetris.DestroyDescriptorSetLayout(d, tetris.compute_set_layouts[0], None)

    for shader, _ in tetris.shader_render:
        tetris.DestroyShaderModule(d, shader, None)

    tetris.DestroyShaderModule(d, tetris.shader_compute, None)
    tetris.DestroyShaderModule(d, tetris.shader_compute_init, None)

    tetris.DestroyCommandPool(d, tetris.command_pool, None)

    tetris.DestroyBuffer(d, tetris.draw_info_buffer, None)
    tetris.DestroyBuffer(d, tetris.attributes_buffer, None)
    tetris.DestroyBuffer(d, tetris.game_data_buffer, None)
    tetris.DestroyBuffer(d, tetris.game_state_buffer, None)
    tetris.FreeMemory(d, tetris.final_alloc, None)
    tetris.FreeMemory(d, tetris.game_state_alloc, None)

    for fence in tetris.fences:
        tetris.DestroyFence(d, fence, None)

    tetris.DestroySemaphore(d, tetris.semaphores[0], None)
    tetris.DestroySemaphore(d, tetris.semaphores[1], None)

    for fb in tetris.framebuffers:
        tetris.DestroyFramebuffer(d, fb, None)

    tetris.DestroyRenderPass(d, tetris.render_pass, None)

    tetris.DestroyImageView(d, tetris.depth_view, None)
    tetris.FreeMemory(d, tetris.depth_memory, None)
    tetris.DestroyImage(d, tetris.depth_image, None)

    for _, view in tetris.swapchain_images:
        tetris.DestroyImageView(d, view, None)

    tetris.DestroySwapchainKHR(d, tetris.swapchain, None)
    tetris.DestroyDevice(d, None)
    tetris.DestroySurfaceKHR(i, tetris.surface, None)

    if DEBUG:
        tetris.DestroyDebugUtilsMessengerEXT(i, tetris.debug_utils, None)

    tetris.DestroyInstance(i, None)


# Shader constants
MAX_OBJECT_COUNT = 64
MAX_INDICES_COUNT = 1000
MAX_ATTRIBUTES_COUNT = 1000
MESH_COUNT = 21
MAX_ASTEROID = 20


# Structures painfully extracted from the shaders
# Should be done automatically by reading the SPIRV binary

class GameObject(Structure):
    _fields_ = (
        ('matrix', c_float*16),
        ('position', c_float*2),
        ('angle', c_float),
        ('velocity', c_float),
        ('display_angle', c_float),
        ('display_angle_update', c_float),
        ('lifetime', c_float),
        ('command_index', c_uint32)
    )


class Mesh(Structure):
    _fields_ = (
        ('indices_offset', c_uint32),
        ('indices_count', c_uint32),
        ('vertex_offset', c_uint32),
        ('vertex_count', c_uint32),
    )

class Asteroid(Structure):
    _fields_ = (
        ('life', c_uint32),
        ('radius', c_uint32),
    ) 

class GameData(Structure):
    _fields_ = (
        ('objects', GameObject * MAX_OBJECT_COUNT),
        ('meshes', Mesh * MAX_OBJECT_COUNT),
        ('asteroids', Asteroid * MAX_ASTEROID),
        ('active_asteroid', c_uint32),
        ('current_level', c_uint32),
        ('asteroidMeshIndex', c_uint32),
        ('asteroidCount', c_uint32),
    )


class GameState(Structure):
    _fields_ = (
        ('game_time', c_double),
        ('past_time', c_double),
        ('up', c_uint32),
        ('left', c_uint32),
        ('right', c_uint32),
        ('space', c_uint32),
        ('reloadTime', c_float),
    )


# Just a data class that will hold the vulkan data.
# I don't want to use globals nor create a hundred of INDENTED object methods
class Tetris(object): pass


def run():
    tetris = Tetris()
    
    create_window(tetris)

    create_instance(tetris)
    create_debug_utils(tetris)
    create_surface(tetris)
    create_device(tetris)

    create_swapchain(tetris)
    create_swapchain_images(tetris)
    create_depth_stencil(tetris)
    create_render_pass(tetris)
    create_framebuffers(tetris)

    create_semaphores(tetris)
    create_fences(tetris)
    create_commands(tetris)
    create_buffers(tetris)
    create_game_state_buffer(tetris)

    create_shaders(tetris)
    create_descriptor_set_layouts(tetris)
    create_descriptors(tetris)
    update_descriptor_sets(tetris)
    create_pipeline_layouts(tetris)
    create_compute_pipeline(tetris)
    create_render_pipeline(tetris)
    create_render_cache(tetris)

    record_commands(tetris)

    run_loop(tetris)

    cleanup(tetris)

    print("It's over")


run()
