# Hardware reference path

The virtual rig models display timing, photodiode edges, contact response, HID polling and clock behavior. The supervisor uses the same serial decoding contract for appropriate physical-reference recordings. The 48-case suite distinguishes measurement acceptance from expected rejection.

`hardware_v133/reference_avr.c` targets ATmega328P at 16 MHz. Optical input is D2/INT0, independent contact feedback is D3/INT1, and the isolated actuator driver is D9. Nominal timer resolution is not measurement accuracy. Use suitable electrical isolation and an independently calibrated timebase. Do not drive a motor directly from a GPIO pin.

`python -m v133.physical capture --help` describes explicit serial capture. Configure actual reference metadata; example or virtual timebase identities are not physical calibration. Firmware build support remains in `hardware_v133/build.py`. A successful compile is separate from loading or validating a physical board.
