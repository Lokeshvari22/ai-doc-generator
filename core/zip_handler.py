import os
import zipfile
import shutil
from config import Config


def extract_zip(zip_path: str) -> tuple:
    """
    Extracts ZIP to EXTRACT_DIR.
    Returns (success, file_count, error_msg)
    """
    # Validate
    if not os.path.exists(zip_path):
        return False, 0, f"File not found: {zip_path}"

    if not zipfile.is_zipfile(zip_path):
        return False, 0, f"Not a valid ZIP: {zip_path}"

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:

            # Test integrity
            bad = z.testzip()
            if bad:
                return False, 0, f"Corrupt file: {bad}"

            # Count source files
            source_files = [
                f for f in z.namelist()
                if any(
                    f.endswith(ext)
                    for ext in Config.SUPPORTED_EXTENSIONS
                )
            ]

            if len(source_files) < Config.MIN_FILES:
                return (False, 0,
                        f"Need min {Config.MIN_FILES} source "
                        f"files, found {len(source_files)}")

            # Clean and extract
            if os.path.exists(Config.EXTRACT_DIR):
                shutil.rmtree(Config.EXTRACT_DIR)
            os.makedirs(Config.EXTRACT_DIR)

            z.extractall(Config.EXTRACT_DIR)

            # Remove macOS junk
            macos = os.path.join(Config.EXTRACT_DIR, "__MACOSX")
            if os.path.exists(macos):
                shutil.rmtree(macos)

        return True, len(source_files), ""

    except zipfile.BadZipFile as e:
        return False, 0, f"Bad ZIP: {e}"
    except Exception as e:
        return False, 0, f"Extraction failed: {e}"