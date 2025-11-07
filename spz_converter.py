import sys
import os
import argparse
from pylogger import PyLogger

from spz_py.ply_loader import load_ply
from spz_py.spz_serializer import serialize_spz

log = PyLogger.getLogger("dcpipeline") # Initialize logger

def load_file(file_path: str) -> dict:
    # Ensure file exists before attempting to open
    if not os.path.exists(file_path):
        log.error(f"Input PLY file not found: {file_path}")
        raise FileNotFoundError(f"Input PLY file not found: {file_path}")
    with open(file_path, 'rb') as f:
        return load_ply(f)

def convert_ply_to_spz(ply_input_path: str, spz_output_path: str):
    """
    Loads a PLY file, converts it to SPZ format, and writes the output.

    Args:
        ply_input_path (str): Path to the input PLY file.
        spz_output_path (str): Path where the output SPZ file will be saved.
    """
    try:
        log.info(f"Loading PLY file from: {ply_input_path}")
        gaussian_data = load_file(ply_input_path)
        log.info(f"Successfully loaded PLY file. {gaussian_data.get('numPoints', 'N/A')} points found.")

        log.info("Converting data to SPZ format...")
        spz_data = serialize_spz(gaussian_data)
        log.info(f"SPZ serialization complete. Output size: {len(spz_data)} bytes.")

        log.info(f"Writing SPZ data to: {spz_output_path}")
        # Ensure output directory exists
        output_dir = os.path.dirname(spz_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            log.info(f"Created output directory: {output_dir}")
        
        with open(spz_output_path, "wb") as f:
            f.write(spz_data)
        log.info(f"Successfully wrote SPZ file to: {spz_output_path}")

    except FileNotFoundError as e:
        # Error already logged in load_file
        sys.exit(1) # Exit with error code
    except Exception as e:
        log.error(f"An error occurred during SPZ conversion: {str(e)}", exc_info=True)
        sys.exit(1) # Exit with error code

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a PLY file to SPZ format.")
    parser.add_argument("ply_input_path", type=str, help="Path to the input PLY file.")
    parser.add_argument("spz_output_path", type=str, help="Path to save the output SPZ file.")

    args = parser.parse_args()

    convert_ply_to_spz(args.ply_input_path, args.spz_output_path)