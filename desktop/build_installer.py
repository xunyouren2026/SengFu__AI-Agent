#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI Unified Framework - PyInstaller Build Script
PyInstaller打包脚本 - 生成独立可执行文件

Usage:
    python build_installer.py          # Build executable
    python build_installer.py --onefile  # Single file mode
    python build_installer.py --clean    # Clean build artifacts

Author: AGI Framework Team
Version: 1.0.0
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path
from typing import List, Optional


class InstallerBuilder:
    """Build standalone installer for AGI Framework"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        self.spec_file = self.project_root / "agi_framework.spec"
        self.is_windows = platform.system() == "Windows"
        self.is_macos = platform.system() == "Darwin"
        self.is_linux = platform.system() == "Linux"
    
    def clean(self):
        """Clean build artifacts"""
        print("🧹 Cleaning build artifacts...")
        for d in [self.build_dir, self.dist_dir]:
            if d.exists():
                shutil.rmtree(d)
                print(f"  Removed: {d}")
        
        # Clean .spec files
        for spec in self.project_root.glob("*.spec"):
            spec.unlink()
            print(f"  Removed: {spec}")
        
        print("✅ Clean complete")
    
    def check_dependencies(self) -> bool:
        """Check if build dependencies are installed"""
        print("🔍 Checking build dependencies...")
        
        missing = []
        try:
            import PyInstaller
            print(f"  ✅ PyInstaller {PyInstaller.__version__}")
        except ImportError:
            missing.append("PyInstaller")
        
        try:
            import pystray
            print(f"  ✅ pystray")
        except ImportError:
            missing.append("pystray")
        
        try:
            from PIL import Image
            print(f"  ✅ Pillow")
        except ImportError:
            missing.append("Pillow")
        
        if missing:
            print(f"\n❌ Missing dependencies: {', '.join(missing)}")
            print(f"   Run: pip install {' '.join(missing)}")
            return False
        
        print("✅ All dependencies satisfied")
        return True
    
    def get_hidden_imports(self) -> List[str]:
        """Get hidden imports for PyInstaller"""
        imports = [
            # Core framework
            "computer_use",
            "computer_use.real_input",
            "computer_use.real_screen",
            "computer_use.real_window",
            "computer_use.agent_brain",
            "computer_use.ocr_engine",
            "computer_use.vision_engine",
            "computer_use.element_engine",
            "computer_use.window_engine",
            "computer_use.clipboard_engine",
            "computer_use.file_ops",
            "computer_use.workflow_recorder",
            "computer_use.monitor",
            "computer_use.screenshot",
            "computer_use.input",
            "computer_use.screen",
            
            # Channel adapters
            "channel",
            "channel.adapters",
            "channel.adapters.dingtalk",
            "channel.adapters.feishu",
            "channel.adapters.slack",
            "channel.adapters.discord",
            "channel.adapters.telegram",
            "channel.adapters.email",
            "channel.adapters.wechat_work",
            "channel.adapters.tuya_iot",
            "channel.adapters.tencent_docs",
            
            # LLM providers
            "llm",
            "llm.providers",
            "llm.providers.openai",
            "llm.providers.deepseek",
            "llm.providers.moonshot",
            "llm.providers.zhipuai",
            "llm.providers.wenxin",
            "llm.providers.dashscope",
            "llm.providers.anthropic",
            
            # Desktop
            "desktop",
            "desktop.system_tray",
            
            # Dependencies
            "aiohttp",
            "pyautogui",
            "pyperclip",
            "mss",
            "PIL",
            "cv2",
            "pytesseract",
            "pystray",
            
            # Async
            "asyncio",
            "json",
            "logging",
            "threading",
            "dataclasses",
            "enum",
            "abc",
            "hashlib",
            "re",
            "time",
            "datetime",
            "pathlib",
            "typing",
            "collections",
        ]
        return imports
    
    def get_data_files(self) -> List[tuple]:
        """Get data files to include"""
        data_files = []
        
        # Web files
        web_dir = self.project_root / "web"
        if web_dir.exists():
            data_files.append((str(web_dir), "web"))
        
        # Config templates
        config_dir = self.project_root / "config" / "templates"
        if config_dir.exists():
            data_files.append((str(config_dir), "config/templates"))
        
        # Channel adapters
        channel_dir = self.project_root / "channel"
        if channel_dir.exists():
            data_files.append((str(channel_dir), "channel"))
        
        # Computer use module
        computer_use_dir = self.project_root / "computer_use"
        if computer_use_dir.exists():
            data_files.append((str(computer_use_dir), "computer_use"))
        
        # Desktop module
        desktop_dir = self.project_root / "desktop"
        if desktop_dir.exists():
            data_files.append((str(desktop_dir), "desktop"))
        
        # LLM module
        llm_dir = self.project_root / "llm"
        if llm_dir.exists():
            data_files.append((str(llm_dir), "llm"))
        
        return data_files
    
    def generate_spec(self, onefile: bool = True):
        """Generate PyInstaller spec file"""
        print("📝 Generating spec file...")
        
        hidden_imports = self.get_hidden_imports()
        data_files = self.get_data_files()
        
        # Format hidden imports
        hiddenimports_str = ",\n        ".join(f'"{imp}"' for imp in hidden_imports)
        
        # Format data files
        datas_str = ""
        for src, dst in data_files:
            datas_str += f'    ("{src}", "{dst}"),\n'
        
        # Platform-specific options
        if self.is_windows:
            console = "False"
            icon = "icon.ico"
        elif self.is_macos:
            console = "False"
            icon = "icon.icns"
        else:
            console = "False"
            icon = "icon.png"
        
        mode = "onefile" if onefile else "onedir"
        
        spec_content = f'''# -*- mode: python ; coding: utf-8 -*-
# AGI Framework PyInstaller Spec
# Auto-generated by build_installer.py

block_cipher = None

a = Analysis(
    ['desktop/system_tray.py'],
    pathex=['{self.project_root}'],
    binaries=[],
    datas=[
{datas_str}    ],
    hiddenimports=[
        {hiddenimports_str},
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'scipy',
        'pandas',
        'IPython',
        'jupyter',
        'notebook',
        'tkinter.test',
        'unittest',
        'pydoc',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AGI_Framework',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console={console},
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='{icon}' if os.path.exists('{icon}') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AGI_Framework',
)

if {onefile}:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='AGI_Framework',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console={console},
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon='{icon}' if os.path.exists('{icon}') else None,
    )
'''
        
        with open(self.spec_file, 'w', encoding='utf-8') as f:
            f.write(spec_content)
        
        print(f"  ✅ Generated: {self.spec_file}")
    
    def build(self, onefile: bool = True, clean_first: bool = False):
        """Build the installer"""
        print(f"\n🚀 Building AGI Framework Installer")
        print(f"   Mode: {'Single File' if onefile else 'Directory'}")
        print(f"   Platform: {platform.system()} {platform.machine()}")
        print()
        
        if clean_first:
            self.clean()
        
        if not self.check_dependencies():
            return False
        
        self.generate_spec(onefile=onefile)
        
        # Run PyInstaller
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--clean",
            "--noconfirm",
            str(self.spec_file)
        ]
        
        print(f"\n📦 Running PyInstaller...")
        print(f"   Command: {' '.join(cmd)}")
        print()
        
        try:
            result = subprocess.run(cmd, cwd=str(self.project_root))
            if result.returncode == 0:
                print(f"\n✅ Build successful!")
                print(f"   Output: {self.dist_dir}")
                
                # List output files
                if self.dist_dir.exists():
                    for item in self.dist_dir.rglob("*"):
                        if item.is_file():
                            size_mb = item.stat().st_size / (1024 * 1024)
                            print(f"   📄 {item.name} ({size_mb:.1f} MB)")
                
                return True
            else:
                print(f"\n❌ Build failed with return code {result.returncode}")
                return False
        except Exception as e:
            print(f"\n❌ Build failed: {e}")
            return False
    
    def create_windows_installer(self):
        """Create Windows MSI installer using Inno Setup"""
        if not self.is_windows:
            print("⚠️  Inno Setup is only available on Windows")
            return False
        
        inno_script = f'''[Setup]
AppName=AGI Framework
AppVersion={1.0}
DefaultDirName={{pf}}\\AGI Framework
DefaultGroupName=AGI Framework
OutputDir=dist
OutputBaseFilename=AGI_Framework_Setup
Compression=lzma2
SolidCompression=yes
SetupIconFile=icon.ico
UninstallDisplayIcon={{app}}\\AGI_Framework.exe
UninstallDisplayName=AGI Framework
LicenseFile=LICENSE
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "dist\\AGI_Framework\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{{group}}\\AGI Framework"; Filename: "{{app}}\\AGI_Framework.exe"
Name: "{{group}}\\Uninstall"; Filename: "{{uninstallexe}}"
Name: "{{commondesktop}}\\AGI Framework"; Filename: "{{app}}\\AGI_Framework.exe"

[Run]
Filename: "{{app}}\\AGI_Framework.exe"; Description: "Launch AGI Framework"; Flags: nowait postinstall shellexec

[Registry]
Root: HKCU; Subkey: "Software\\AGI Framework"; ValueType: string; ValueName: "InstallPath"; ValueData: "{{app}}"
'''
        
        inno_path = self.project_root / "installer.iss"
        with open(inno_path, 'w', encoding='utf-8') as f:
            f.write(inno_script)
        
        print(f"  ✅ Generated Inno Setup script: {inno_path}")
        print(f"  ℹ️  Run with Inno Setup Compiler to create installer")
        return True


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description="AGI Framework Installer Builder")
    parser.add_argument("--onefile", action="store_true", default=True, help="Build as single file")
    parser.add_argument("--onedir", action="store_true", help="Build as directory")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--installer", action="store_true", help="Create Windows installer")
    parser.add_argument("--check", action="store_true", help="Check dependencies only")
    args = parser.parse_args()
    
    builder = InstallerBuilder()
    
    if args.clean:
        builder.clean()
        return
    
    if args.check:
        builder.check_dependencies()
        return
    
    onefile = not args.onedir
    success = builder.build(onefile=onefile)
    
    if success and args.installer:
        builder.create_windows_installer()


if __name__ == "__main__":
    main()
