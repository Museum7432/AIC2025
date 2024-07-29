import streamlit as st
from st_clickable_images import clickable_images

import utils
from typing import List, Tuple
from PIL import Image
import os 
import json

def extrac_name_file_img(key_fr_id,video_name):
    if int(video_name[1:3])>=17:
        l=3-len(key_fr_id)
        return '0'*l+key_fr_id
    else:
        l=4-len(key_fr_id)
        return '0'*l+key_fr_id



def read_image(results: List[dict]) -> List[Image.Image]:
    images = []
    image_names = []
    IMAGE_KEYFRAME_PATH = "/app/KeyFrames"  # Đường dẫn đến thư mục chứa keyframes
    
       

    results= sorted(results, key=lambda k: k['score'])
    for res in results:
       

        video_name = res["video_name"]
        folder_name = "keyframes_"+ video_name[:3]
     
       
        video_folder=os.path.join(IMAGE_KEYFRAME_PATH,folder_name, video_name)
        
        img_name=extrac_name_file_img(str(res['keyframe_id']),res['video_name'])+'.jpg'

        image_path = os.path.join(video_folder,img_name)
        image = Image.open(image_path)

        image_names.append( video_name + " "+ img_name)
        images.append(image)

        
        # keyframe_id= int(keyframe_id)
        
        # if os.path.exists(video_folder):
        #     image_files = sorted(os.listdir(video_folder))


        #     if keyframe_id < len(image_files):

        #         image_file = image_files[keyframe_id]
        #         image_path = os.path.join(video_folder, image_file)
          
        #         image = Image.open(image_path)

        #         image_names.append( video_name + " "+ image_file)
        #         images.append(image)
        #     else:
        #         print(f"Keyframe id {keyframe_id} is out of range for video {video_name}.")

    return images, image_names


def main() -> None:
    st.set_page_config(layout="wide")
    st.title("ELO@AIC Image Semantic Search")
    #Specific_folder= st.text_input("Video folder", value="", placeholder="e.g L01_V006", key = 1)
 
    #if st.button("Check folder"): 
        #images_List ,image_names = utils.get_images(Specific_folder )
        #utils.Print_Images(images_List ,image_names)

    query, k , model_choice = utils.start_sidebar()

    if not query:

        return 
    


    st.subheader("Query Results")
    img_result = utils.send_request(query, k, model_choice)
    

    
    images_List ,image_names  = read_image(img_result['search_result'])

    # with st.container():
    #     for col in st.columns(5):
    #         col.image(images_List, caption= image_names ,  width=150)
   
    
    num_columns = 3  # Number of columns in the grid
    col_count = len(images_List)


    # Loop through the image_paths and display each image in a column
    for i in range(0, col_count, num_columns):
        columns = st.columns(num_columns)  # Create columns for the grid

        for j in range(num_columns):
            idx = i + j
            if idx < col_count:
                with columns[j]:  # Use columns[j] to display images in each column
                    st.image(images_List[idx], use_column_width=True, caption=image_names[idx])


  
    

   
    
if __name__ == "__main__":
    main()
