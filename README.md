# HandConductor
A small program that allows the user to change the color of a TP-LINK LB130 light bulb to preset colors by holding up their fingers

This project uses the mediapipe hands library to determine the key landmarks on the user's hands that are shown in the frame. The landmark data is used to determine the number of fingers that are currently held up. Then a thread is created to run an npm command that will change the current color state of the light bulb based on the number of fingers.

This program is not currently set up to be easily ported. It is configured to run using a video stream that is locally hosted from my raspberry pi

libraries:
https://github.com/konsumer/tplink-lightbulb
https://google.github.io/mediapipe/solutions/hands.html


Example of program running:
https://user-images.githubusercontent.com/44955732/173703289-6b036170-3af7-4fec-bfb3-561fab09d02d.mp4

