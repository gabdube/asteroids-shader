from enum import Enum
from collections import namedtuple


class EventsMap(dict):

    def __init__(self):
        self.key_press = []
        self.key_release = []

    def __iter__(self):
        events = tuple(self.keys())
        for e in events:
            data = self.get(e, CORRUPTED)
            if data is not CORRUPTED:
                yield e, data

        yield from self.key_press
        yield from self.key_release

        self.key_press.clear()
        self.key_release.clear()

        self.clear()

    def __setitem__(self, event, event_data):
        if event is KeyPress:
            self.key_press.append((event, event_data))
        elif event is KeyRelease:
            self.key_release.append((event, event_data))
        elif event in Events:
            super().__setitem__(event, event_data)
        else:
            raise KeyError(f"Invalid event type: {key}")


# Might be returned if an event was not processed corrrectly. Things like that might happens if the process
# is locked by IO and some events that are called many times (ex: MouseMove) are executed at the same time
CORRUPTED = namedtuple("Corrupted", ())

Events = Enum("Events", "WindowResized RenderEnable RenderDisable MouseClick MouseMove MouseScroll KeyPress KeyRelease")

WindowResized = Events.WindowResized
WindowResizedData = namedtuple('WindowResizedData', 'width height')

RenderEnable = Events.RenderEnable
RenderDisable = Events.RenderDisable

MouseClick = Events.MouseClick
MouseClickState = Enum("MouseButtonState", "Down Up")
MouseClickButton = Enum("MouseClickButton", "Left Right Middle")
MouseClickData = namedtuple("MouseClickData", "state button x y")

MouseMove = Events.MouseMove
MouseMoveData = namedtuple("MouseMoveData", "x y")

MouseScroll = Events.MouseScroll
MouseScrollData = namedtuple("MouseScrollData", "delta x y")

KeyPress = Events.KeyPress
KeyPressData = namedtuple("KeyPressData", "key")

KeyRelease = Events.KeyRelease
KeyReleaseData = namedtuple("KeyReleaseData", "key")

keys_values = "Back Tab Clear Return Shift Control Menu Pause Capital Kana Junja Final Hanja Kanji Escape Convert Space " \
"Left Up Right Down " \
"_0 _1 _2 _3 _4 _5 _6 _7 _8 _9"
Keys = Enum("Keys", keys_values)

ArrowKeys = (Keys.Left, Keys.Up, Keys.Right, Keys.Down)
NumKeys = (Keys._0, Keys._1, Keys._2, Keys._3, Keys._4, Keys._5, Keys._6, Keys._8, Keys._9)
