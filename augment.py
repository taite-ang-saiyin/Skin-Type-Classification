import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

input_dir = "config/valid/Acne prone skin"  
output_dir = "config/valid/Acne prone skin"  

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


datagen = ImageDataGenerator(
    rotation_range=30,        
    width_shift_range=0.2,    
    height_shift_range=0.2,   
    shear_range=0.2,          
    zoom_range=0.2,           
    horizontal_flip=True,     
    fill_mode='nearest'       
)


augmented_image_count = 0
total_augmented_images_needed = 40

for filename in os.listdir(input_dir):
    aug_count=0
    img_path = os.path.join(input_dir, filename)
    
   
    img = load_img(img_path)
    x = img_to_array(img) 
    x = x.reshape((1,) + x.shape)  

    for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix='damage', save_format='jpeg'):
        augmented_image_count += 1
        aug_count+=1
        if aug_count > 3:
            break

    if augmented_image_count >= total_augmented_images_needed:
        break

print(f"Total augmented images created: {augmented_image_count}")
