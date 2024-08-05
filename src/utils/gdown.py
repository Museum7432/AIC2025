import os
import gdown

def download_folder(folder_id, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of files in the Google Drive folder
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url,output_folder=output_folder)
    # Move and rename the downloaded folder
    downloaded_folder = folder_id  # gdown creates a folder with the folder_id as its name
    if os.path.exists(downloaded_folder):
        for item in os.listdir(downloaded_folder):
            source = os.path.join(downloaded_folder, item)
            destination = os.path.join(output_folder, item)
            
            if os.path.isfile(source):
                if os.path.exists(destination):
                    print(f"Skipping existing file: {item}")
                    os.remove(source)
                else:
                    os.rename(source, destination)
                    print(f"Downloaded: {item}")
            elif os.path.isdir(source):
                if not os.path.exists(destination):
                    os.rename(source, destination)
                    print(f"Downloaded folder: {item}")
                else:
                    print(f"Skipping existing folder: {item}")

        # Remove the empty folder
        os.rmdir(downloaded_folder)


if __name__ == "__main__":
    folder_id = "1GglRr3SGso65a6IOxhk7gDi5sbTbfPm_"
    output_folder = "dataset/zip_keyframes"
    
    download_folder(folder_id, output_folder)

    