# -*- coding: utf-8 -*-

"""
    Minimalistic wrapper over the XCB window api
"""
import weakref
from ctypes import *
from time import perf_counter
from .. import events as e


# Extern libraries
try:
    xcb = cdll.LoadLibrary('libxcb.so.1')
    libc = cdll.LoadLibrary('libc.so.6')
except OSError:
    raise OSError('Failed to import libxcb.so or libc. Are they installed?')

# TYPES
xcb_connection_t = POINTER(c_void_p)
xcb_window_t = c_uint
xcb_colormap_t = c_uint
xcb_visualid_t = c_uint
xcb_drawable_t = c_uint
xcb_atom_t = c_uint
xcb_timestamp_t = c_uint

# Structures
class xcb_setup_t(Structure):
    _fields_ = (
        ('status', c_ubyte),
        ('pad0', c_ubyte),
        ('protocol_major_version', c_uint16),
        ('protocol_minor_version', c_uint16),
        ('length', c_uint16),
        ('release_number', c_uint32),
        ('resource_id_base', c_uint32),
        ('resource_id_mask', c_uint32),
        ('motion_buffer_size', c_uint32),
        ('vendor_len', c_uint16),
        ('maximum_request_length', c_uint16),
        ('roots_len', c_ubyte),
        ('pixmap_formats_len', c_ubyte),
        ('image_byte_order', c_ubyte),
        ('bitmap_format_bit_order', c_ubyte),
        ('bitmap_format_scanline_unit', c_ubyte),
        ('bitmap_format_scanline_pad', c_ubyte),
        ('min_keycode', c_ubyte),
        ('max_keycode', c_ubyte),
        ('pad1', c_ubyte*4)
    )

class xcb_screen_t(Structure):
    _fields_ = (
        ('root', xcb_window_t),
        ('default_colormap', xcb_colormap_t),
        ('white_pixel', c_uint),
        ('black_pixel', c_uint),
        ('current_input_mask', c_uint),
        ('width_in_pixels', c_ushort), ('height_in_pixels', c_ushort),
        ('width_in_millimeters', c_ushort), ('height_in_millimeters', c_ushort),
        ('min_installed_maps', c_ushort), ('max_installed_maps', c_ushort),
        ('root_visual', xcb_visualid_t),
        ('backing_stores', c_ubyte), ('save_unders', c_ubyte),
        ('root_depth', c_ubyte), ('allowed_depths_len', c_ubyte)
    )

class xcb_motion_notify_event_t(Structure):
    _fields_ = (
        ('response_type', c_ubyte),
        ('detail', c_ubyte),
        ('sequence', c_ushort),
        ('time', xcb_timestamp_t),
        ('root', xcb_window_t),
        ('event', xcb_window_t),
        ('child', xcb_window_t),
        ('root_x', c_short), ('root_y', c_short),
        ('event_x', c_short), ('event_y', c_short),
        ('state', c_ushort),
        ('same_screen', c_ubyte),
        ('pad0', c_ubyte)
    )

xcb_button_press_event_t = xcb_motion_notify_event_t

class xcb_key_press_event_t(Structure):
    _fields_ = (
        ('response_type', c_ubyte),
        ('detail', c_ubyte),
        ('sequence', c_ushort),
        ('time', xcb_timestamp_t),
        ('root', xcb_window_t),
        ('event', xcb_window_t),
        ('child', xcb_window_t),
        ('root_x', c_short), ('root_y', c_short),
        ('event_x', c_short), ('event_y', c_short),
        ('state', c_ushort),
        ('same_screen', c_ubyte),
        ('pad0', c_ubyte)
    )

class xcb_configure_notify_event_t(Structure):
    _fields_ = (
        ('response_type', c_ubyte),
        ('pad0', c_ubyte),
        ('sequence', c_ushort),
        ('event', xcb_window_t),
        ('window', xcb_window_t),
        ('above_sibling', xcb_window_t),
        ('x', c_short), ('y', c_short),
        ('width', c_short), ('height', c_ushort),
        ('border_width', c_ushort),
        ('override_redirect', c_ubyte),
        ('pad1', c_ubyte)
    )

class xcb_get_geometry_reply_t(Structure):
    _fields_ = (
        ('response_type', c_ubyte),
        ('depth', c_ubyte),
        ('sequence', c_ushort),
        ('length', c_uint),
        ('root', xcb_window_t),
        ('x', c_short), ('y', c_short),
        ('width', c_ushort), ('height', c_ushort),
        ('border_width', c_ushort)
    )

class xcb_generic_event_t(Structure):
    _fields_ = (
        ('response_type', c_ubyte),
        ('pad0', c_ubyte),
        ('sequence', c_ushort),
        ('pad', c_uint*7),
        ('full_sequence', c_uint),
    )

class xcb_intern_atom_reply_t(Structure):
    _fields_ = (
        ('response_type', c_ubyte),
        ('pad0', c_ubyte),
        ('sequence', c_ushort),
        ('length', c_uint),
        ('atom', xcb_atom_t),
    )

class xcb_screen_iterator_t(Structure):
    _fields_ = (
        ('data', POINTER(xcb_screen_t)),
        ('rem', c_int),
        ('index', c_int)
    )

class xcb_void_cookie_t(Structure):
    _fields_ = (('sequence', c_uint),)

class xcb_get_keyboard_mapping_reply_t(Structure):
    _fields_ = (
        ('response_type', c_uint8),
        ('keysyms_per_keycode', c_uint8),
        ('sequence', c_uint16),
        ('length', c_uint32),
        ('pad0', c_uint8 * 24)
    )

xcb_get_geometry_cookie_t = xcb_void_cookie_t
xcb_intern_atom_cookie_t = xcb_void_cookie_t
xcb_get_keyboard_mapping_cookie_t = xcb_void_cookie_t

# CONSTS

NULL = c_void_p(0)
NULL_STR = cast(NULL, POINTER(c_char))

XCB_CW_BACK_PIXEL = 2
XCB_CW_EVENT_MASK = 2048

XCB_EVENT_MASK_KEY_PRESS = 1
XCB_EVENT_MASK_KEY_RELEASE = 2
XCB_EVENT_MASK_EXPOSURE = 32768
XCB_EVENT_MASK_STRUCTURE_NOTIFY = 131072
XCB_EVENT_MASK_POINTER_MOTION = 64
XCB_EVENT_MASK_BUTTON_PRESS = 4
XCB_EVENT_MASK_BUTTON_RELEASE = 8

XCB_COPY_FROM_PARENT = 0

XCB_PROP_MODE_REPLACE = 0

XCB_ATOM_STRING = 31
XCB_ATOM_WM_NAME = 39

XCB_WINDOW_CLASS_INPUT_OUTPUT = 1

XCB_KEY_PRESS = 2
XCB_KEY_RELEASE = 3
XCB_BUTTON_PRESS = 4
XCB_BUTTON_RELEASE = 5
XCB_MOTION_NOTIFY = 6
XCB_DESTROY_NOTIFY = 17
XCB_CLIENT_MESSAGE = 33
XCB_CONFIGURE_NOTIFY = 22

XCB_BUTTON_INDEX_1 = 1
XCB_BUTTON_INDEX_2 = 2
XCB_BUTTON_INDEX_3 = 3
XCB_BUTTON_INDEX_4 = 4
XCB_BUTTON_INDEX_5 = 5

XCB_CONFIG_WINDOW_X = 1
XCB_CONFIG_WINDOW_Y = 2

XK_0 = 0x0030
XK_1 = 0x0031
XK_2 = 0x0032
XK_3 = 0x0033
XK_4 = 0x0034
XK_5 = 0x0035
XK_6 = 0x0036
XK_7 = 0x0037
XK_8 = 0x0038
XK_9 = 0x0039

XK_Left = 0xff51
XK_Up = 0xff52
XK_Right = 0xff53
XK_Down = 0xff54

XK_Space = 0x0020

k = e.Keys
key_map = {
    XK_Space: k.Space,
    XK_Left: k.Left,
    XK_Up: k.Up,
    XK_Right: k.Right,
    XK_Down: k.Down,
    XK_0: k._0,
    XK_1: k._1,
    XK_2: k._2,
    XK_3: k._3,
    XK_4: k._4,
    XK_5: k._5,
    XK_6: k._6,
    XK_7: k._7,
    XK_8: k._8,
    XK_9: k._9,
}

# Functions

xcb_connect = xcb.xcb_connect
xcb_connect.restype = xcb_connection_t
xcb_connect.argtypes = (c_char_p, POINTER(c_int))

xcb_get_setup = xcb.xcb_get_setup
xcb_get_setup.restype = POINTER(xcb_setup_t)
xcb_get_setup.argtypes = (xcb_connection_t,)

xcb_setup_roots_iterator = xcb.xcb_setup_roots_iterator
xcb_setup_roots_iterator.restype = xcb_screen_iterator_t
xcb_setup_roots_iterator.argtypes = (POINTER(xcb_setup_t),)

xcb_screen_next = xcb.xcb_screen_next
xcb_screen_next.restype = None
xcb_screen_next.argtypes = (POINTER(xcb_screen_iterator_t),)

xcb_disconnect = xcb.xcb_disconnect
xcb_disconnect.restype = None
xcb_disconnect.argtypes = (xcb_connection_t,)

xcb_generate_id = xcb.xcb_generate_id
xcb_generate_id.restype = c_uint
xcb_generate_id.argtypes = (xcb_connection_t, )

xcb_create_window = xcb.xcb_create_window
xcb_create_window.restype = xcb_void_cookie_t
xcb_create_window.argtypes = (
    xcb_connection_t, c_ubyte, xcb_window_t, xcb_window_t, c_short, c_short,
    c_short, c_short, c_short, c_short, xcb_visualid_t, c_uint, c_void_p
)

xcb_map_window = xcb.xcb_map_window
xcb_map_window.restype = xcb_void_cookie_t
xcb_map_window.argtypes = (xcb_connection_t, xcb_window_t)

xcb_unmap_window = xcb.xcb_unmap_window
xcb_unmap_window.restype = xcb_void_cookie_t
xcb_unmap_window.argtypes = (xcb_connection_t, xcb_window_t)

xcb_destroy_window = xcb.xcb_destroy_window
xcb_destroy_window.restype = xcb_void_cookie_t
xcb_destroy_window.argtypes = (xcb_connection_t, xcb_window_t)

xcb_flush = xcb.xcb_flush
xcb_flush.restype = c_int
xcb_flush.argtypes = (xcb_connection_t,)

xcb_get_geometry = xcb.xcb_get_geometry
xcb_get_geometry.restype = xcb_get_geometry_cookie_t
xcb_get_geometry.argtypes = (xcb_connection_t, xcb_drawable_t)

xcb_get_geometry_reply = xcb.xcb_get_geometry_reply
xcb_get_geometry_reply.restype = POINTER(xcb_get_geometry_reply_t)
xcb_get_geometry_reply.argtypes = (xcb_connection_t, xcb_get_geometry_cookie_t, c_void_p)

xcb_poll_for_event = xcb.xcb_poll_for_event
xcb_poll_for_event.restype = POINTER(xcb_generic_event_t)
xcb_poll_for_event.argtypes = (xcb_connection_t,)

xcb_intern_atom = xcb.xcb_intern_atom
xcb_intern_atom.restype = xcb_intern_atom_cookie_t
xcb_intern_atom.argtypes = (xcb_connection_t, c_ubyte, c_ushort, c_char_p)

xcb_intern_atom_reply = xcb.xcb_intern_atom_reply
xcb_intern_atom_reply.restype = POINTER(xcb_intern_atom_reply_t)
xcb_intern_atom_reply.argtypes = (xcb_connection_t, xcb_intern_atom_cookie_t , c_void_p)

xcb_change_property = xcb.xcb_change_property
xcb_change_property.restype = xcb_void_cookie_t
xcb_change_property.argtypes = (
    xcb_connection_t, c_ubyte, xcb_window_t, xcb_atom_t, xcb_atom_t,
    c_ubyte, c_uint, c_void_p
)

xcb_get_keyboard_mapping = xcb.xcb_get_keyboard_mapping
xcb_get_keyboard_mapping.restype = xcb_get_keyboard_mapping_cookie_t
xcb_get_keyboard_mapping.argtypes = (xcb_connection_t, c_ubyte, c_ubyte)

xcb_get_keyboard_mapping_reply = xcb.xcb_get_keyboard_mapping_reply
xcb_get_keyboard_mapping_reply.restype = POINTER(xcb_get_keyboard_mapping_reply_t)
xcb_get_keyboard_mapping_reply.argtypes = (xcb_connection_t, xcb_get_keyboard_mapping_cookie_t, c_void_p)

free = libc.free
free.restype = None
free.argtypes = (c_void_p,)


class XcbWindow(object):
    
    def __init__(self, **kwargs):
        __allowed_kwargs = ('width', 'height', 'fixed')
        bad_kwargs_keys = [k for k in kwargs.keys() if k not in __allowed_kwargs]
        if len(bad_kwargs_keys) != 0 and type(self) is XcbWindow:
            raise AttributeError("Some unknown keyword were found: {}".format(','.join(bad_kwargs_keys)))

        # Setup a window using XCB
        screen_count = c_int(0)
        connection = xcb_connect(NULL_STR, byref(screen_count))

        setup = xcb_get_setup(connection)

        iter = xcb_setup_roots_iterator(setup)
        screen_count, screens = iter.rem, []
        while screen_count != 0:
            if iter.data:
                screens.append(iter.data.contents)
            screen_count -= 1

            xcb_screen_next(byref(iter))

        _screen = screens[0]

        # Create the window
        events_masks = XCB_EVENT_MASK_BUTTON_RELEASE | XCB_EVENT_MASK_BUTTON_PRESS |\
                        XCB_EVENT_MASK_POINTER_MOTION | XCB_EVENT_MASK_STRUCTURE_NOTIFY |\
                        XCB_EVENT_MASK_EXPOSURE | XCB_EVENT_MASK_KEY_RELEASE | XCB_EVENT_MASK_KEY_PRESS

        window = xcb_generate_id(connection)
        value_mask = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK
        value_list = (c_uint*32)(_screen.black_pixel, events_masks)
      
        full_width, full_height = _screen.height_in_pixels, _screen.width_in_pixels
        width, height = kwargs.get('width', 1280), kwargs.get('height', 720)
        x, y = (full_width//2) - (width//2), (full_height//2) - (height//2)

        xcb_create_window(
            connection, XCB_COPY_FROM_PARENT, window, _screen.root,
            0, 0, width, height, 0, XCB_WINDOW_CLASS_INPUT_OUTPUT,
            _screen.root_visual, value_mask,
            cast(value_list, c_void_p)
        )

        # Magic code that will send notification when window is destroyed
        cookie = xcb_intern_atom(connection, 1, 12, b'WM_PROTOCOLS')
        reply = xcb_intern_atom_reply(connection, cookie, NULL)

        cookie2 = xcb_intern_atom(connection, 0, 16, b'WM_DELETE_WINDOW')
        atom_wm_delete_window = xcb_intern_atom_reply(connection, cookie2, 0)

        xcb_change_property(
            connection,
            XCB_PROP_MODE_REPLACE,
            window,
            reply.contents.atom,
            4, 32, 1,
            byref(xcb_atom_t(atom_wm_delete_window.contents.atom))
        )

        free(reply)

        # Save the required members and start listening to user events
        self.__window = window
        self.__connection = connection
        self.__setup = setup
        self.__position = (y, x)
        self.cached_size = (width, height)
        self.events = e.EventsMap()
        self.must_exit = False

        self.resizing = False

        self.set_title("VulkanTechDemo")

        self.internal_keymaps = {}
        self.map_key_symbols()
        

    def destroy(self):
        if self.__connection is None or self.__window is None:
            return

        xcb_destroy_window(self.__connection, self.__window)
        xcb_disconnect(self.__connection)
        self.__connection = None
        self.__window = None

    @property
    def handle(self):
        return self.__window

    @property
    def connection(self):
        return self.__connection

    def set_title(self, title):
        title = title.encode()
        len_title = len(title)
        ctitle = c_char_p(title)

        xcb_change_property(
            self.__connection,
            XCB_PROP_MODE_REPLACE,
		    self.__window, XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8,
		    len_title, ctitle
        )

    def show(self):
        xcb_map_window(self.__connection, self.__window)
        self.set_position(*self.__position)
        xcb_flush(self.__connection)

    def hide(self):
        xcb_unmap_window(self.__connection, self.__window)
        xcb_flush(self.__connection)
        
    def set_position(self, x, y):
        values = (c_uint*2)(x, y)
        xcb.xcb_configure_window(
            self.__connection,
            self.__window,
            XCB_CONFIG_WINDOW_X|XCB_CONFIG_WINDOW_Y,
            byref(values)
        )
        xcb_flush(self.__connection)
        self.__position = (x, y)

    def dimensions(self):
        return self.cached_size

    def handle_event(self, event_ptr):
        event = event_ptr.contents
        evt_id = event.response_type & 0x7f

        if evt_id == XCB_MOTION_NOTIFY:
            motion_event = cast(event_ptr, POINTER(xcb_motion_notify_event_t)).contents
            x, y = float(motion_event.event_x), float(motion_event.event_y)
            self.events[e.MouseMove] = e.MouseMoveData(x, y)

        elif evt_id == XCB_CONFIGURE_NOTIFY:
            resize_event = cast(event_ptr, POINTER(xcb_configure_notify_event_t)).contents
            width, height = resize_event.width, resize_event.height
            c_width, c_height = self.cached_size
            if width != c_width or height != c_height:
                self.cached_size = (width, height)
                self.resizing = perf_counter()
                self.events[e.RenderDisable] = None

        elif evt_id == XCB_BUTTON_PRESS or evt_id == XCB_BUTTON_RELEASE:
            press_event = cast(event_ptr, POINTER(xcb_button_press_event_t)).contents

            mstate = e.MouseClickState
            if evt_id == XCB_BUTTON_PRESS:
                state = mstate.Down
            elif evt_id == XCB_BUTTON_RELEASE:
                state = mstate.Up

            sys_button, mbtn = press_event.detail, e.MouseClickButton
            if sys_button == XCB_BUTTON_INDEX_1:
                button = mbtn.Left
            elif sys_button == XCB_BUTTON_INDEX_3:
                button = mbtn.Right
            elif sys_button == XCB_BUTTON_INDEX_2:
                button = mbtn.Middle
            elif sys_button == XCB_BUTTON_INDEX_4 or sys_button == XCB_BUTTON_INDEX_5:
                # Mouse wheel scroll
                delta = 1 if sys_button == XCB_BUTTON_INDEX_4 else -1
                self.events[e.MouseScroll] = e.MouseScrollData(
                    delta = delta, 
                    x = press_event.event_x,
                    y = press_event.event_y
                )
                return
            else:
                # Unimplemented buttons / return
                return

            self.events[e.MouseClick] = e.MouseClickData(
                state = state,
                button = button,
                x = press_event.event_x, y = press_event.event_y
            )

        elif evt_id == XCB_KEY_PRESS or evt_id == XCB_KEY_RELEASE:
            key_event = cast(event_ptr, POINTER(xcb_key_press_event_t)).contents
            key_sym_id = self.internal_keymaps[key_event.detail]

            event_key = key_map.get(key_sym_id)
            if event_key is None:
                return

            if evt_id == XCB_KEY_PRESS:
                self.events[e.KeyPress] = e.KeyPressData(
                    key = event_key
                )
            else:
                self.events[e.KeyRelease] = e.KeyReleaseData(
                    key = event_key
                )

        elif evt_id in (XCB_CLIENT_MESSAGE, XCB_DESTROY_NOTIFY):
            self.must_exit = True

    def poll_status(self):
        if self.resizing:
            c, rc = perf_counter(), self.resizing
            if c-rc > 0.3:
                self.events[e.WindowResized] = e.WindowResizedData(*self.cached_size)
                self.events[e.RenderEnable] = None
                self.resizing = False

    def translate_system_events(self):
        conn, handle_events = self.__connection, self.handle_event

        self.poll_status()

        event_ptr = xcb_poll_for_event(conn)
        while event_ptr:
            self.handle_event(event_ptr)
            event_ptr = xcb_poll_for_event(conn)

    def map_key_symbols(self):
        conn, setup = self.__connection, self.__setup.contents

        mapping_cookie = xcb_get_keyboard_mapping(conn, setup.min_keycode, setup.max_keycode - setup.min_keycode + 1)
        mapping_ptr = xcb_get_keyboard_mapping_reply(conn, mapping_cookie, None)
        mapping = mapping_ptr.contents

        keysyms_per_keycode = mapping.keysyms_per_keycode
        symbols_count = mapping.length
        keycodes_count = symbols_count // keysyms_per_keycode

        symbols_array = c_uint32 * symbols_count
        symbols_array_address = cast(mapping_ptr, c_void_p).value + sizeof(xcb_get_keyboard_mapping_reply_t)
        symbols = symbols_array.from_address(symbols_array_address)  

        for key_code_index in range(keycodes_count):
            key_code_real = setup.min_keycode + key_code_index
            self.internal_keymaps[key_code_real] = symbols[key_code_index * keysyms_per_keycode]

        free(mapping_ptr)
