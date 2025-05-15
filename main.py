# main.py

from timbre_transfer import TimbreTransfer
import os

# Predefined configuration
INPUT_FILE       = 'DDSP Reaper/DDSP Reaper_stems_Viloin.wav' 
MODEL_FILE  = 'models/Guitar_model.h5'  
OUTPUT_FILE      = 'out/viloin_2_guit_2.wav'  


def main():
    # Ensure output directory exists
    out_dir = os.path.dirname(OUTPUT_FILE)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"Converting '{INPUT_FILE}' → Guitar timbre...")
    transfer = TimbreTransfer(MODEL_FILE)
    transfer.transfer_file(INPUT_FILE, OUTPUT_FILE)
    print(f"Done! Guitar-style audio written to '{OUTPUT_FILE}'")


if __name__ == '__main__':
    main()
