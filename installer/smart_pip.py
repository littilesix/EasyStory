import os
import sys
import argparse
import subprocess
import tempfile
import importlib.util
from importlib.metadata import distribution, PackageNotFoundError
from importlib import invalidate_caches
import inspect

def set_file_env():
    caller_frame = inspect.stack()[1]
    caller_filepath = os.path.abspath(caller_frame.filename)
    caller_dir = os.path.dirname(caller_filepath)
    libs = os.path.join(caller_dir, "site-packages")
    if libs not in sys.path:
        sys.path.insert(0, libs)
    return caller_dir, libs

def is_installed(pkg_name):
    try:
        distribution(pkg_name)  # e.g. 'huggingface-hub'
        return True
    except PackageNotFoundError:
        return False

def parse_wheel_name(wheel_path):
    """Extract package name from the wheel file name."""
    base = os.path.basename(wheel_path)
    return base.split("-")[0].replace("_", "-").lower()

def smart_install(package_name, target_dir, add_path=False,upgrade=None):
    """
    Install the specified package and its dependencies to target_dir,
    skipping packages already installed in the current environment.

    Args:
        package_name (str): Name of the main package to install.
        target_dir (str): Directory to install packages into.
        add_path (bool): If True, temporarily add target_dir to sys.path.

    Returns:
        bool: True if installation succeeded or no install needed, False if error.
    """
    try:
        target_dir = os.path.abspath(target_dir)

        if add_path:
            sys.path.insert(0, target_dir)
            print(f"📌 Temporarily added {target_dir} to PYTHONPATH")

        os.makedirs(target_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"🔍 Downloading {package_name} and its dependencies to temporary directory...")
            for f in os.listdir(tmpdir):os.remove(os.path.join(tmpdir,f))
            subprocess.run([
                sys.executable, "-m", "pip", "download", package_name,
                "--dest", tmpdir,
                "-i","https://mirrors.aliyun.com/pypi/simple/"
            ], check=True)

            wheels = [f for f in os.listdir(tmpdir) if f.endswith(".whl")]

            missing_packages = []

            for wheel in wheels:
                pkg_name = parse_wheel_name(wheel)

                if not is_installed(pkg_name):
                    print(f"⏬ Package needed: {pkg_name} {is_installed(pkg_name)}")
                    missing_packages.append(os.path.join(tmpdir, wheel))
                else:
                    print(f"✔ Package already installed: {pkg_name}, skipping")

            print(missing_packages)
            if missing_packages:
                print(f"📦 Installing missing packages to {target_dir}")
                subprocess.run([
                    sys.executable, "-m", "pip", "install",
                    "--no-deps", "--target", target_dir,
                ] + missing_packages, check=True)
            else:
                print("✅ All dependencies are satisfied, no installation needed")

        invalidate_caches()
        
        return True

    except Exception as e:
        print(f"❌ Error during installation: {e}")
        return False
    

def main():
    parser = argparse.ArgumentParser(description="Smart pip installer: install missing dependencies only to a target directory")
    parser.add_argument("--package", required=True, help="Main package name to install")
    parser.add_argument("--target", required=True, help="Target directory to install packages")
    parser.add_argument("--add-path", action="store_true", help="Temporarily add the target directory to PYTHONPATH for the current process")

    args = parser.parse_args()

    success = smart_install(args.package, args.target, args.add_path)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
    #print(is_installed("huggingface-hub"))
