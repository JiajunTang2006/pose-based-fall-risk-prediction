# App Icon Assets

Put the user-designed FallGuard icon here as:

```text
apps/macos/assets/FallGuard.png
```

Recommended source size: `1024x1024` PNG.

The desktop GUI loads this PNG at startup when it exists. The macOS build script converts it into:

```text
apps/macos/assets/FallGuard.icns
```

The script first tries `iconutil`; if that fails, it falls back to `tiff2icns`.

Do not edit generated icon files directly. Update `FallGuard.png` and rerun `apps/macos/build_app.sh`.
