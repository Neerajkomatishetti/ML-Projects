from manim import *
import numpy as np

# Make sure to install Pillow and SciPy:
#   pip install pillow scipy

class ConvolutionPipelineAnimation(Scene):
    def construct(self):
        ##############################
        # Stage 1: Load and Display the User’s Input Image
        ##############################
        # Replace the path below with the actual image file on your system.
        image_path = "4.png"  # <-- Provide your image file here!
        input_img = ImageMobject(image_path)
        # Scale the image for display (adjust as needed).
        input_img.scale_to_fit_height(4)
        input_img.to_edge(LEFT, buff=0.5)
        input_title = Text("Input Image", font_size=24)
        input_title.next_to(input_img, UP)
        self.play(FadeIn(input_img), Write(input_title))
        self.wait(1)
        
        ##############################
        # Stage 2: Animate the Convolution Kernel Sliding Over the Image
        ##############################
        # For demonstration, we overlay a rectangle (the kernel window) on the image.
        # (In a real convolution each pixel would be used; here we show a few steps.)
        kernel_window = Rectangle(width=1, height=1, color=YELLOW)
        # Position the kernel at a chosen starting location on the input image.
        # We use one of the image’s corners (with a small offset) for demonstration.
        kernel_start = input_img.get_corner(UP + LEFT) + 0.5 * RIGHT + 0.5 * DOWN
        kernel_window.move_to(kernel_start)
        self.play(Create(kernel_window))
        self.wait(0.5)
        
        # Define several positions (in display coordinates) to simulate sliding.
        slide_positions = [
            kernel_start,
            kernel_start + RIGHT * 0.8,
            kernel_start + RIGHT * 0.8 + DOWN * 0.8,
            kernel_start + DOWN * 0.8,
        ]
        for pos in slide_positions:
            self.play(kernel_window.animate.move_to(pos), run_time=0.8)
            self.wait(0.5)
        
        ##############################
        # Stage 3: Compute the Convolution on the Actual Image
        ##############################
        # We now process the actual image (using Pillow and SciPy) to compute a feature map.
        # Import the required libraries.
        from PIL import Image, ImageOps
        import scipy.signal
        
        # Open the image and convert to grayscale.
        pil_img = Image.open(image_path).convert("L")
        img_arr = np.array(pil_img, dtype=np.float32)
        
        # Define a convolution kernel.
        # Here we use an edge detection kernel.
        conv_kernel = np.array([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]], dtype=np.float32)
        
        # Perform a 2D convolution (using 'valid' mode so that only fully overlapping patches are used).
        convolved_arr = scipy.signal.convolve2d(img_arr, conv_kernel, mode='valid')
        
        # Normalize the result to the [0, 255] range.
        convolved_arr = convolved_arr - convolved_arr.min()
        if convolved_arr.max() != 0:
            convolved_arr = convolved_arr / convolved_arr.max() * 255
        convolved_arr = convolved_arr.astype(np.uint8)
        
        # Save the convolved result as an image file.
        convolved_pil = Image.fromarray(convolved_arr)
        output_image_path = "conv_images/convolved_image.png"
        convolved_pil.save(output_image_path)
        
        # Create a Manim ImageMobject from the convolved image.
        convolved_img = ImageMobject(output_image_path)
        convolved_img.scale_to_fit_height(4)
        convolved_img.to_edge(RIGHT, buff=0.5)
        output_title = Text("Convolved Feature Map", font_size=24)
        output_title.next_to(convolved_img, UP)
        
        # Transition: Show the convolved feature map.
        self.play(FadeIn(convolved_img), Write(output_title))
        self.wait(1)
        
        ##############################
        # Stage 4: Flatten the Feature Map and Feed into an MLP
        ##############################
        # For many networks, the feature map is flattened into a vector.
        # For demonstration we extract a row of pixel values from the convolved array.
        center_row = convolved_arr[convolved_arr.shape[0] // 2]
        # Sample 10 values evenly from this row.
        sample_indices = np.linspace(0, len(center_row) - 1, num=10, dtype=int)
        sample_values = center_row[sample_indices]
        
        flattened_vector = VGroup(*[
            MathTex(str(round(val, 1)), font_size=24)
            for val in sample_values
        ])
        flattened_vector.arrange(RIGHT, buff=0.3)
        flattened_vector.next_to(convolved_img, DOWN, buff=0.5)
        flatten_label = Text("Flattened Vector", font_size=24)
        flatten_label.next_to(flattened_vector, DOWN)
        self.play(Write(flattened_vector), Write(flatten_label))
        self.wait(1)
        
        # Represent the MLP as a purple box.
        mlp_box = RoundedRectangle(corner_radius=0.2, height=2, width=3, color=PURPLE)
        mlp_label = Text("MLP", font_size=28, color=PURPLE)
        mlp_group = VGroup(mlp_box, mlp_label)
        mlp_group.next_to(flattened_vector, RIGHT, buff=1)
        self.play(Create(mlp_box), Write(mlp_label))
        self.wait(0.5)
        
        # Draw an arrow from the flattened vector to the MLP.
        arrow = Arrow(flattened_vector.get_right(), mlp_box.get_left(), buff=0.1, color=WHITE)
        self.play(Create(arrow))
        self.wait(0.5)
        
        # Finally, display the MLP’s output.
        mlp_output = MathTex("y", font_size=36, color=ORANGE)
        mlp_output.next_to(mlp_box, RIGHT, buff=0.5)
        self.play(Write(mlp_output))
        self.wait(2)



