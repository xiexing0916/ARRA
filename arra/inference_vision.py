from inference_solver import FlexARInferenceSolver
from PIL import Image
import os


# ******************* Image Generation ******************
inference_solver = FlexARInferenceSolver(
    model_path=None,  # Replace with the directory of model
    precision="bf16",
    target_size=512,
)

output_dir = './generated_image'  # Replace with the directory where resized images will be saved


prompt = ""  # Given the prompt
output_dir = output_dir.replace("generated_image", prompt)
print(output_dir)
os.makedirs(output_dir, exist_ok=True)

for i in range(5):
    output_path = os.path.join(output_dir, f"image_{i+1}.png")

    qas = [[prompt, None]]

    try:
        generated2 = inference_solver.generate(
            images=[],
            qas=qas,
            max_gen_len=8192,
            temperature=1.0,
            logits_processor=inference_solver.create_logits_processor(cfg=4.0, image_top_k=2000),
        )
        if generated2[1][0] is not None:
            a2, new_image = generated2[0], generated2[1][0]
            print(f"Generated text for image {i+1}: {a2}")
            if isinstance(new_image, Image.Image):
                new_image.save(output_path)
                print(f"Image {i+1} saved to {output_path}")
            else:
                print(f"Generated output for image {i+1} is not a valid image.")
        else:
            a2 = generated2[0]
            print(f"Generated text for image {i+1}: {a2}")
    except Exception as e:
        print(f"An error occurred during generation for image {i+1}: {e}")

