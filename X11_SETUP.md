# X11 Forwarding Setup for WSL2

This guide will help you set up X11 forwarding so you can see GUI applications from WSL2 on your Windows desktop.

## Step 1: Install VcXsrv (X Server for Windows)

1. Download VcXsrv from: https://sourceforge.net/projects/vcxsrv/
2. Install it (default settings are fine)
3. Launch **XLaunch** (it should be in your Start menu)

## Step 2: Configure VcXsrv

When XLaunch starts, configure it as follows:

1. **Display settings**: Choose "Multiple windows" (or "One large window" if you prefer)
2. **Client startup**: Choose "Start no client"
3. **Extra settings**:

   - ✅ Check "Disable access control" (IMPORTANT - allows WSL2 to connect)
   - ✅ Check "Native opengl" (optional, for better performance)
   - ✅ Check "Primary Selection" (optional, for clipboard sharing)

4. Click "Finish" to start the X server

## Step 3: Verify Setup in WSL2

The `.zshrc` file has been updated to automatically set the `DISPLAY` variable.

**Reload your shell configuration:**

```bash
source ~/.zshrc
```

**Or start a new terminal session.**

**Test if it works:**

```bash
echo $DISPLAY
# Should show something like: 172.19.16.1:0.0
```

## Step 4: Test with a Simple GUI App

Install a test tool (optional):

```bash
sudo apt-get update
sudo apt-get install -y x11-apps
```

Test it:

```bash
xeyes
# or
xclock
```

If you see a window appear on Windows, X11 forwarding is working! 🎉

## Step 5: Run Your Traffic Simulator

Now you can run your traffic simulator with GUI:

```bash
cd /home/neely/dev/traffic-monitor-sim
source venv/bin/activate
python traffic_sim.py
```

## Troubleshooting

### If you get "Cannot connect to X server":

1. Make sure VcXsrv is running on Windows
2. Check that "Disable access control" is enabled in VcXsrv settings
3. Verify DISPLAY variable: `echo $DISPLAY`
4. Try restarting VcXsrv

### If DISPLAY is not set automatically:

Manually set it:

```bash
export DISPLAY=$(ip route show | grep default | awk '{print $3}'):0.0
```

### If the IP changes (WSL2 restart):

The `.zshrc` configuration will automatically detect the new IP each time you start a shell.

## Alternative: X410 (Paid)

If you prefer a paid solution with better integration:

- X410 from Microsoft Store (~$10)
- Better Windows integration
- Automatic DISPLAY configuration
