# Blender Viewport Trackpad Navigation Implementation (Windows)

This document details how Blender implements precision trackpad/touchpad navigation in the viewport on Windows.

## Overview

Blender uses **Microsoft's DirectManipulation API** (a COM-based high-level gesture recognition framework) rather than handling raw WM_POINTER messages. This provides hardware-accelerated gesture recognition, automatic smoothing, and inertia support.

---

## 1. Windows API Usage

### DirectManipulation API

**Primary Location:** `intern/ghost/intern/GHOST_TrackpadWin32.cc`

Blender uses the following DirectManipulation COM interfaces:

| Interface | Purpose |
|-----------|---------|
| `IDirectManipulationManager` | Main coordinator for gesture recognition |
| `IDirectManipulationViewport` | Viewport context for gesture tracking |
| `IDirectManipulationUpdateManager` | Drives gesture transform state updates |
| `IDirectManipulationViewportEventHandler` | Callback handler for gesture events |

**Location:** `intern/ghost/intern/GHOST_TrackpadWin32.cc` (lines 62-67)

Calls viewport->ActivateConfiguration with the flags:
- DIRECTMANIPULATION_CONFIGURATION_INTERACTION
- DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_X
- DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_Y
- DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_INERTIA
- DIRECTMANIPULATION_CONFIGURATION_SCALING

### Configuration Flags Explained

| Flag | Purpose |
|------|---------|
| `INTERACTION` | Enable basic interaction mode |
| `TRANSLATION_X` | Enable horizontal panning |
| `TRANSLATION_Y` | Enable vertical panning |
| `TRANSLATION_INERTIA` | Enable momentum after pan gesture ends |
| `SCALING` | Enable pinch-to-zoom |

### Pointer Type Detection

When a `DM_POINTERHITTEST` message arrives, Blender calls `GetPointerType()` to check for `PT_TOUCHPAD` specifically, distinguishing trackpad input from other pointer sources.

### Why DirectManipulation Instead of WM_POINTER?

DirectManipulation provides:
- Automatic gesture recognition (pan vs. pinch detection)
- Hardware-accelerated processing
- Built-in inertia/momentum handling
- Noise filtering and smoothing

The trade-off is less low-level control, but better consistency across different trackpad hardware.

---

## 2. GHOST Layer Implementation

GHOST (Generic Handy Operating System Toolkit) is Blender's window system abstraction layer.

### Key Classes

**`GHOST_DirectManipulationHelper`** (`intern/ghost/intern/GHOST_TrackpadWin32.cc`)

The class provides four main methods:
- A static factory method that creates a helper instance for a given window handle and DPI setting
- An update method that drives DirectManipulation frame updates
- A pointer hit test handler that registers pointer contact with DirectManipulation
- A getter method that retrieves accumulated gesture data (pan deltas, scale, scroll direction)

**`GHOST_DirectManipulationViewportEventHandler`**

Implements the `IDirectManipulationViewportEventHandler` COM interface with callbacks:
- `OnViewportStatusChanged()` - Detects gesture start/end transitions
- `OnViewportUpdated()` - Viewport state changes
- `OnContentUpdated()` - Receives gesture transformation matrix updates

### Gesture State Machine

The gesture state is tracked using three possible states: none (idle), pan, or pinch. The state machine automatically transitions from PAN to PINCH when scaling is detected (lines 324-346 in GHOST_TrackpadWin32.cc).

---

## 3. Event Types and Data Structures

### Trackpad Event Subtypes

**Location:** `intern/ghost/GHOST_Types.h`

The trackpad event subtype enumeration defines six event types:
- Unknown (default/uninitialized)
- Scroll (two-finger pan/scroll)
- Rotate (reserved, not currently used)
- Swipe (reserved, not currently used)
- Magnify (pinch zoom)
- SmartMagnify (double-tap magnify)

### Event Data Structure

The trackpad event data structure contains:
- The event subtype (scroll, magnify, etc.)
- Current cursor X and Y positions (32-bit integers)
- Delta X value (pan delta, or scale factor for pinch gestures)
- Delta Y value (pan delta)
- A flag indicating whether the system's scroll direction is inverted (natural scrolling)

### Internal Trackpad Info Structure

The internal trackpad info structure stores:
- Accumulated pan deltas for X and Y axes (32-bit integers)
- Accumulated scale factor (32-bit integer)
- A boolean indicating the Windows scroll direction setting (natural scrolling preference)

---

## 4. Window Configuration and Initialization

### Window Setup

**Location:** `intern/ghost/intern/GHOST_WindowWin32.cc`

During window initialization, Blender performs two key setup steps:
1. At line 170, it registers the window for touch input using the Windows API
2. At line 220, it creates the DirectManipulation helper instance, passing the window handle and current DPI setting for proper scaling

### DirectManipulation Viewport Configuration

**Location:** `intern/ghost/intern/GHOST_TrackpadWin32.cc` (lines 62-76)

The DirectManipulation viewport is configured with a combination of flags that enable:
- Basic interaction mode
- Horizontal translation (X-axis panning)
- Vertical translation (Y-axis panning)
- Translation inertia (momentum after releasing)
- Scaling (pinch-to-zoom)

Key configuration choices:
- **MANUALUPDATE mode** (line 75): Blender drives DirectManipulation manually each frame, not through DirectComposition
- **Artificial viewport rect**: 10000x10000 pixels (used only for gesture normalization, not actual content)
- **Inertia enabled**: Momentum continuation after gesture ends

### Precision Touchpad Registry Integration

**Location:** `intern/ghost/intern/GHOST_TrackpadWin32.cc` (lines 118-168)

The implementation reads the user's scroll direction preference from the Windows registry at the path `HKEY_CURRENT_USER\SOFTWARE\Microsoft\Windows\CurrentVersion\PrecisionTouchPad\ScrollDirection`. It also registers for real-time change notifications on this registry key, allowing Blender to respect the system "natural scrolling" preference without requiring a restart.

---

## 5. Event Flow: From Hardware to Viewport Navigation

### Complete Pipeline

1. **Pointer Contact Detection** (`GHOST_SystemWin32.cc:2389`)
   - Windows sends `DM_POINTERHITTEST` when pointer touches trackpad
   - Handler: `GHOST_WindowWin32::onPointerHitTest()`
   - Action: Registers pointer with DirectManipulation via `SetContact(pointerId)`

2. **DirectManipulation Processing** (`GHOST_TrackpadWin32.cc`)
   - DirectManipulation recognizes gestures internally
   - Fires `OnContentUpdated()` callback with transformation matrix
   - Extracts pan deltas from `transform[4]`, `transform[5]`
   - Extracts scale from `transform[0]`

3. **Event Loop Polling** (`GHOST_SystemWin32.cc:451, 462`)
   - Each frame: `driveTrackpad()` updates DirectManipulation context
   - Each frame: `processTrackpad()` retrieves accumulated gesture values

4. **GHOST Event Creation** (`GHOST_SystemWin32.cc:1604-1639`)
   - Pan gestures → `GHOST_kTrackpadEventScroll` → `MOUSEPAN` event type
   - Pinch gestures → `GHOST_kTrackpadEventMagnify` → `MOUSEZOOM` event type

5. **Window Manager Processing** (`source/blender/windowmanager/intern/wm_event_system.cc:6034-6085`)
   - Maps GHOST trackpad events to internal event types
   - `MOUSEPAN` → Pan navigation
   - `MOUSEZOOM` → Zoom navigation

6. **Viewport Navigation Operators** (`source/blender/editors/space_view3d/`)
   - `VIEW3D_OT_move` receives `MOUSEPAN` events
   - `VIEW3D_OT_zoom` receives `MOUSEZOOM` events
   - Movement calculated as: `2 * event->xy[0] - event->prev_xy[0]` (view3d_navigate_view_move.cc:95)

---

## 6. Two-Finger Gesture Handling (Precision Navigation)

### Pan (Scroll) Gesture

Two fingers moving together in the same direction:

1. DirectManipulation tracks as `TRANSLATION_X` and `TRANSLATION_Y`
2. The X and Y translation values are extracted from the transformation matrix (indices 4 and 5) and divided by the device scale factor for DPI-aware handling
3. The floating-point values are converted to integer deltas, but the fractional parts are preserved across frames to maintain sub-pixel precision (lines 334-341)

### Pinch (Magnify) Gesture

Two fingers moving apart or together:

1. DirectManipulation tracks as `SCALING`
2. The scale value is extracted from the transformation matrix (index 0) and multiplied by a pinch scale factor of 125.0 to normalize the gesture sensitivity
3. State machine transitions from PAN to PINCH automatically (lines 310-314)

### Noise Filtering

**Location:** `GHOST_TrackpadWin32.cc:292-300`

An epsilon threshold of 0.00003 is used to filter out extremely small delta values. If the absolute value of a delta is below this threshold, the update is skipped entirely. This prevents micro-movements from generating excessive events.

### Gesture Reset

The viewport reset function performs the following steps:
1. If a gesture is currently active, it resets the DirectManipulation viewport to its original 10000x10000 pixel rectangle
2. Sets the gesture state back to idle (none)
3. Resets the last scale value to the default pinch scale factor
4. Clears the last X and Y position values to zero

---

## 7. Key Technical Details

### Event Accumulation and Deduplication

**Location:** `wm_event_system.cc:5791-5797`

Trackpad events are accumulated - only the latest state is sent per frame. This prevents event queue flooding during fast gestures.

### Fractional Precision Preservation

Floating-point gesture data is converted to integers, but fractional parts are accumulated across frames. This maintains sub-pixel precision for smooth navigation.

### DPI Awareness

All trackpad calculations consider the display's DPI scaling factor (line 286) to maintain consistent behavior across different monitor configurations.

### Multi-Touch Feature Flag

Trackpad events are only processed if `system->multitouch_gestures_` is true (line 2393). This is a user-configurable preference.

### Driver Compatibility Workaround

**Location:** `GHOST_SystemWin32.cc:91-101`

A compile-time flag is defined to handle broken touchpad drivers. Some laptop touchpad drivers have messaging issues where updates are not properly delivered. The workaround posts a dummy user message to force the Windows `PeekMessage()` function to return updates correctly.

---

## 8. File Reference

| File | Purpose |
|------|---------|
| `intern/ghost/intern/GHOST_TrackpadWin32.hh` | DirectManipulation helper class declaration |
| `intern/ghost/intern/GHOST_TrackpadWin32.cc` | DirectManipulation implementation |
| `intern/ghost/intern/GHOST_WindowWin32.cc` | Window creation, touch registration |
| `intern/ghost/intern/GHOST_SystemWin32.cc` | Event loop, trackpad polling |
| `intern/ghost/GHOST_Types.h` | Event type definitions |
| `intern/ghost/intern/GHOST_EventTrackpad.hh` | Trackpad event class |
| `source/blender/windowmanager/intern/wm_event_system.cc` | Event mapping to operators |
| `source/blender/editors/space_view3d/view3d_navigate_view_move.cc` | Pan operator |
| `source/blender/editors/space_view3d/view3d_navigate_view_zoom.cc` | Zoom operator |

---

## 9. DirectManipulation Implementation Details (FAQ)

### Q1: Does Blender call EnableMouseInPointer(TRUE)?

**No.** Blender does not call `EnableMouseInPointer()` anywhere in the codebase. DirectManipulation works without it.

### Q2: What window styles does Blender use when creating its HWND?

**Location:** `intern/ghost/intern/GHOST_WindowWin32.cc` (lines 87-96, 109)

**Window styles (WS_*):**
| Window Type | Styles |
|-------------|--------|
| Top-level window | `WS_OVERLAPPEDWINDOW` |
| Child window (with parent) | `WS_POPUPWINDOW \| WS_CAPTION \| WS_MAXIMIZEBOX \| WS_MINIMIZEBOX \| WS_SIZEBOX` |
| Fullscreen | Above styles `\| WS_MAXIMIZE` |

**Extended window styles (WS_EX_*):**
| Window Type | Extended Styles |
|-------------|-----------------|
| Top-level window | `0` (none) |
| Child window (with parent) | `WS_EX_APPWINDOW` (forces onto taskbar, allows minimization) |

Note: Dialog windows are mentioned in comments but not fully implemented; they would use `WS_POPUPWINDOW | WS_CAPTION` with `WS_EX_DLGMODALFRAME | WS_EX_TOPMOST`.

### Q3: What are the exact parameters to RegisterHitTestTarget and when is it called?

**Blender does NOT call RegisterHitTestTarget.** Instead, Blender relies on Windows automatically sending `DM_POINTERHITTEST` messages when pointer input is first detected. When these messages arrive, Blender calls `IDirectManipulationViewport::SetContact(pointerId)` to register the pointer contact with the DirectManipulation viewport.

**Hit test handling flow:**
1. Windows sends `DM_POINTERHITTEST` message to the window
2. `GHOST_SystemWin32::s_wndProc()` receives the message (line 2389)
3. Calls `GHOST_WindowWin32::onPointerHitTest(wParam)` (lines 235-248)
4. Checks if pointer type is `PT_TOUCHPAD` via `GetPointerType()`
5. If touchpad, calls `direct_manipulation_viewport_->SetContact(pointerId)` (line 172 in GHOST_TrackpadWin32.cc)

### Q4: Does Blender call RegisterTouchWindow or any other API to enable pointer/touch input?

**Yes.** Blender calls `RegisterTouchWindow()` at line 170 in `GHOST_WindowWin32.cc`, passing the window handle and a flags value of zero.

**Parameters:**
- `hwnd`: The window handle
- `ulFlags`: `0` (no special flags - not using `TWF_FINETOUCH` or `TWF_WANTPALM`)

**Timing:** Called after `setDrawingContextType()` succeeds, before `RegisterDragDrop()`.

No other touch/pointer registration APIs are called (no `EnableMouseInPointer`, no `RegisterPointerInputTarget`, etc.).

### Q5: Is there any special window class registration with touch-related flags?

**No touch-related flags.** The window class uses standard redraw flags only.

**Location:** `intern/ghost/intern/GHOST_SystemWin32.cc` (lines 643-660)

The window class is registered with a `WNDCLASSW` structure containing:
- **Style flags**: `CS_HREDRAW | CS_VREDRAW` (redraw on horizontal/vertical resize)
- **Window procedure**: The static `s_wndProc` message handler
- **Extra bytes**: Zero for both class and window extra storage
- **Instance handle**: Retrieved via `GetModuleHandle(0)`
- **Icon**: Loaded from the "APPICON" resource
- **Cursor**: Standard arrow cursor (`IDC_ARROW`)
- **Background brush**: Dark gray stock brush
- **Menu name**: None (null)
- **Class name**: `L"GHOST_WindowClass"`

Only `CS_HREDRAW | CS_VREDRAW` are used - no `CS_TOUCH_*` flags or other special touch/pointer flags.

### Q6: What is the complete DirectManipulation initialization order?

**Location:** `intern/ghost/intern/GHOST_TrackpadWin32.cc` (lines 35-115)

The complete initialization sequence in `GHOST_DirectManipulationHelper::create()`:

| Step | Action | Line |
|------|--------|------|
| 1 | Create the DirectManipulation manager COM object as an in-process server | 46-49 |
| 2 | Retrieve the update manager interface from the manager | 54 |
| 3 | Create a viewport for the window handle (with null frame info) | 58-59 |
| 4 | Activate the viewport configuration with interaction mode, X/Y translation, translation inertia, and scaling enabled | 62-69 |
| 5 | Set viewport options to manual update mode | 74-75 |
| 6 | Add the event handler to the viewport, receiving a cookie for later removal | 84-85 |
| 7 | Set the viewport rectangle to a 10000x10000 pixel area starting at origin | 89-90 |
| 8 | Activate the manager for the window handle | 93 |
| 9 | Enable the viewport | 96 |
| 10 | Reset the viewport state via the event handler | 99 |

**Key observations:**
- **No RegisterHitTestTarget call** - Blender relies on Windows automatically routing pointer hit test messages
- Configuration activation is called before setting viewport options
- The viewport rectangle is set before activating the manager (viewport must be configured before activation)
- The manager is activated before the viewport is enabled
- All this happens **after** the window is created and visible (window shown at line 208, DirectManipulation initialized at line 220)

### Q7: Does Blender do anything special with COM initialization beyond standard CoInitializeEx?

**Blender uses `OleInitialize()` instead of `CoInitializeEx()`.**

**Location:** `intern/ghost/intern/GHOST_SystemWin32.cc` (line 183)

At line 183, Blender calls `OleInitialize(0)` with a null parameter. This is done specifically to support the drop target functionality that will be created later in `GHOST_WindowWin32`.

**Why OleInitialize?**
- `OleInitialize(0)` internally calls `CoInitialize()` with STA (single-threaded apartment) semantics
- It also initializes OLE (Object Linking and Embedding) support, which is required for:
  - `GHOST_DropTargetWin32` (drag-and-drop functionality)
  - `RegisterDragDrop()` API

**Additional COM usage:**
- In `GHOST_SystemPathsWin32.cc:137`: `CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE)` is used separately for shell integration (recent files list)
- DirectManipulation COM objects are created with `CLSCTX_INPROC_SERVER` (in-process server)

**Timing:** `OleInitialize()` is called during `GHOST_SystemWin32` construction, before any windows are created. This ensures COM/OLE is ready for:
1. DirectManipulation (`CoCreateInstance` for the manager)
2. Drag-and-drop (`RegisterDragDrop`)
3. Taskbar progress (`CoCreateInstance` for `ITaskbarList3`)

### Q8: Does Blender fallback to WM_MOUSEWHEEL if DirectManipulation is unavailable?

**Yes, but without inertia/smoothing.** Blender has separate handlers for `WM_MOUSEWHEEL` and `WM_MOUSEHWHEEL` that operate independently from DirectManipulation.

**Location:** `intern/ghost/intern/GHOST_SystemWin32.cc` (lines 1266-1315, 2064-2082)

**How the wheel handler works:**

The `processWheelEventVertical()` function handles vertical wheel events:
1. Retrieves the current accumulated delta value for the wheel axis
2. Extracts the new wheel delta from the message using `GET_WHEEL_DELTA_WPARAM()`
3. If the scroll direction reverses (current accumulator and new delta have opposite signs), resets the accumulator to zero
4. Adds the new delta to the accumulator
5. For each `WHEEL_DELTA` (120 units) worth of accumulated movement, pushes a discrete `GHOST_EventWheel` event with direction +1 or -1
6. Preserves the remainder (fractional wheel ticks) for the next message

**Key differences from DirectManipulation trackpad events:**

| Feature | DirectManipulation (Trackpad) | WM_MOUSEWHEEL (Legacy) |
|---------|-------------------------------|------------------------|
| Event type | `GHOST_kTrackpadEventScroll` → `MOUSEPAN` | `GHOST_EventWheel` → discrete wheel ticks |
| Inertia | Built-in momentum after gesture ends | None |
| Smoothing | Hardware-accelerated noise filtering | None |
| Precision | Sub-pixel floating-point deltas | Integer direction (+1/-1) per `WHEEL_DELTA` |
| Accumulation | Sub-pixel remainder preserved | Sub-tick remainder preserved |

**Conclusion:** The `WM_MOUSEWHEEL` handler does NOT apply inertia or smoothing. It only preserves fractional wheel ticks across messages (sub-`WHEEL_DELTA` accumulation). If a touchpad is reporting as a legacy mouse device, the user will get discrete stepped scrolling, not the smooth trackpad feel from DirectManipulation.

### Q9: What is the exact content of Blender's Application Manifest?

**Location:** `release/windows/manifest/blender.exe.manifest.in`

Blender's application manifest is a standard Windows 10/8.1 manifest with no special touch, pointer, or input-related flags.

**Complete manifest structure:**

| Section | Content |
|---------|---------|
| Trust Info | Requests `asInvoker` execution level with `uiAccess="false"` |
| Window Settings | Sets `activeCodePage` to UTF-8 for proper Unicode file path handling |
| Compatibility | Declares support for Windows 10 (`{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}`) and Windows 8.1 (`{1f676c76-80e1-4239-95bb-83d0f6d0da78}`) |
| Dependencies | Requires Common Controls v6.0 (`Microsoft.Windows.Common-Controls`) |

**Key observations:**

- **No `maxVersionTested`**: The manifest does not include a `maxVersionTested` element, which is sometimes used to opt into newer Windows behaviors
- **Standard OS GUIDs**: Uses the standard Microsoft-documented GUIDs for Windows 10 and 8.1 compatibility
- **No touch-specific settings**: No `disableWindowFiltering`, `gdiScaling`, `heapType`, or pointer input settings
- **No DPI awareness declaration**: DPI awareness is likely handled via code (`SetProcessDpiAwareness`) rather than manifest

**Conclusion:** Blender's manifest does not include any special compatibility flags that would influence how Windows routes pointer/touch input. The `PT_TOUCHPAD` pointer type detection relies entirely on the Windows Precision Touchpad driver properly identifying the hardware, not on manifest settings.

### Q10: Does Blender use RegisterRawInputDevices to bypass mouse emulation?

**No.** Blender does not call `RegisterRawInputDevices()` anywhere in the codebase for mouse or touchpad input.

A codebase-wide search confirms that this API is not used for pointer devices. Blender does not bypass the Windows Pointer/Mouse emulation layer via Raw Input for touchpad/mouse input.

**What Blender does use for raw input:**

Blender uses the Raw Input API for **keyboard** and optionally **NDOF (3D mouse)** input, not for mouse/touchpad:

**Location:** `intern/ghost/intern/GHOST_SystemWin32.cc` (lines 132-161)

The `initRawInput()` function registers raw input devices during system initialization:
- Registers 1-2 `RAWINPUTDEVICE` structures (keyboard always, NDOF if compiled with `WITH_INPUT_NDOF`)
- Keyboard device: `usUsagePage = 0x01` (Generic Desktop Controls), `usUsage = 0x06` (Keyboard)
- NDOF device (optional): `usUsagePage = 0x01`, `usUsage = 0x08` (Multi-axis Controller)
- No `dwFlags` set (defaults to 0, meaning normal input sink behavior)

This raw input registration is used for:
- Retrieving true key presses (not affected by keyboard layout translation)
- Detecting left vs. right modifier key presses (Shift, Ctrl, Alt)
- SpaceNavigator and other 3D mice support (when compiled with NDOF support)

**Conclusion:** Blender relies entirely on the standard Windows message pipeline for mouse and touchpad input:
- **Precision Touchpad**: `DM_POINTERHITTEST` → DirectManipulation → trackpad events
- **Legacy touchpad/mouse**: `WM_MOUSEWHEEL`/`WM_MOUSEHWHEEL` → discrete wheel events
- **Mouse movement**: `WM_MOUSEMOVE` → cursor events

Blender does not use Raw Input to bypass driver-level mouse emulation for touchpads.
