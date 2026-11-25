VC TRANSLATOR - get the .exe with NO installs on your PC
=========================================================
GitHub builds the program for you, for free, on its own Windows computers.

1. Make a free account at github.com, then click New repository.
   Name it vc-translator, choose PUBLIC, tick "Add a README", Create.
2. In the repository: Add file -> Upload files. Open this unzipped folder and drag
   EVERYTHING inside it (including the hidden-looking ".github" folder) into the page.
   Click "Commit changes".
3. Open the "Actions" tab. A job called "Build installer" starts by itself.
   Wait about 15-25 minutes until it shows a green tick.
4. Open "Releases" (right side of the repo page) and download VCTranslator-Setup.exe.
   Run it. Done.

UPDATES (no uninstall)
- Edit a file on GitHub (or upload a new one), and raise the number in version.py
  (1.0.0 -> 1.0.1). GitHub rebuilds and publishes a new release automatically.
- The installed program checks for updates on start-up (or press "Check for updates")
  and installs the new version over the old one, then restarts.
- Your settings and the downloaded models/GPU files stay in %APPDATA%\VCTranslator.

FIRST START
- The first Start downloads the speech model (and ~1 GB of NVIDIA files if you use the GPU). One time only.
- For "App: ..." capture: start your game, join voice chat, press the refresh button, pick the app.

Prefer building on your own PC? Install Python 3.12 and Inno Setup 6, then double-click build.bat.
