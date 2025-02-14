from manim import *
import numpy as np
from PIL import Image
import scipy.signal

class AdvancedConvolutionAnimation(Scene):
    def construct(self):
        ##############################
        # Stage 1: Load and Display the User’s Input Image
        ##############################
        # Replace this with the actual file name/path of your image.
        image_path = "five.jpg"
        # Open the image with Pillow a
        # nd convert it to grayscale for processing.
        pil_img = Image.open(image_path).convert("L")
        input_arr = np.array(pil_img, dtype=np.float32)

        # For display, we convert the grayscale image back to RGB.
        pil_img_rgb = pil_img.convert("RGB")
        input_img = ImageMobject(pil_img_rgb)
        input_img.scale_to_fit_height(4)
        input_img.to_edge(LEFT, buff=0.5)
        input_label = Text("Input Image", font_size=24)
        input_label.next_to(input_img, UP)
        self.play(FadeIn(input_img), Write(input_label))
        self.wait(1)

        ##############################
        # Stage 2: Define Kernels & Process Convolutions
        ##############################
        # Define three different 3x3 kernels.
        # 1. Heatmap kernel (a smooth, “warm” filter)
        kernel_heat = np.array([[0, 1, 0],
                                [1, 2, 1],
                                [0, 1, 0]], dtype=np.float32)
        kernel_heat = kernel_heat / kernel_heat.sum()  # normalize

        # 2. Edge detection kernel
        kernel_edge = np.array([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]], dtype=np.float32)

        # 3. Sharpen kernel
        kernel_sharpen = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]], dtype=np.float32)

        # Store kernels in a dictionary along with a color for the sliding-window.
        kernels = {
            "Heatmap": {"kernel": kernel_heat, "color": YELLOW},
            "Edge": {"kernel": kernel_edge, "color": RED},
            "Sharpen": {"kernel": kernel_sharpen, "color": GREEN}
        }

        # Dictionary to hold the first-level convolution output ImageMobjects.
        first_conv_outputs = {}

        # For each kernel, perform the following:
        for i, (name, info) in enumerate(kernels.items()):
            kernel = info["kernel"]
            color = info["color"]

            # Compute the convolution on the input image using SciPy.
            conv_result = scipy.signal.convolve2d(input_arr, kernel, mode='valid')
            # Normalize the result to the range 0-255.
            conv_result = conv_result - conv_result.min()
            if conv_result.max() != 0:
                conv_result = conv_result / conv_result.max() * 255
            conv_result = conv_result.astype(np.uint8)

            # Save the result temporarily so we can load it as an ImageMobject.
            temp_filename = f"temp_output_{name}.png"
            Image.fromarray(conv_result).save(temp_filename)
            conv_mobject = ImageMobject(temp_filename)
            conv_mobject.scale_to_fit_height(3)
            # Position the output on the right side, staggering vertically.
            conv_mobject.to_edge(RIGHT, buff=0.5)
            conv_mobject.shift(UP * (2 - i * 3.5))
            conv_label = Text(f"{name} Output", font_size=24)
            conv_label.next_to(conv_mobject, UP)

            # Animate a sliding kernel on the input image.
            # For demonstration, we draw a rectangle representing the 3x3 patch.
            sliding_rect = Rectangle(
                width=input_img.get_width() / 5,
                height=input_img.get_height() / 5,
                color=color
            )
            # Start at the top–left corner (with a small offset).
            start_pos = input_img.get_corner(UL) + RIGHT * sliding_rect.get_width() / 2 + DOWN * sliding_rect.get_height() / 2
            sliding_rect.move_to(start_pos)
            self.play(Create(sliding_rect))
            self.wait(0.5)
            # Simulate sliding by moving the rectangle a few times.
            slide_positions = [
                start_pos,
                start_pos + RIGHT * 0.5,
                start_pos + RIGHT * 0.5 + DOWN * 0.5
            ]
            for pos in slide_positions:
                self.play(sliding_rect.animate.move_to(pos), run_time=0.8)
                self.wait(0.3)
            self.play(FadeOut(sliding_rect))
            self.wait(0.5)

            # Reveal the computed convolution output concurrently.
            self.play(FadeIn(conv_mobject), Write(conv_label))
            self.wait(1)

            first_conv_outputs[name] = conv_mobject

        ##############################
        # Stage 3: Stack All First-Level Feature Maps
        ##############################
        outputs_list = list(first_conv_outputs.values())
        # Use Group instead of VGroup to avoid type errors with ImageMobject.
        stacked_outputs = Group(*outputs_list).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        stacked_outputs.to_edge(RIGHT, buff=0.5)
        stack_label = Text("Stacked Feature Maps", font_size=24)
        stack_label.next_to(stacked_outputs, UP)
        self.play(Write(stack_label))
        self.wait(1)

        ##############################
        # Stage 4: Apply a Second Convolution on Each Feature Map
        ##############################
        # Define a second convolution kernel (here an averaging filter).
        second_kernel = np.array([[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]], dtype=np.float32) / 9.
        second_conv_outputs = {}
        second_output_mobjects = []
        for i, (name, first_out) in enumerate(first_conv_outputs.items()):
            # Open the first output image from file.
            temp_filename = f"temp_output_{name}.png"
            pil_first = Image.open(temp_filename).convert("L")
            first_arr = np.array(pil_first, dtype=np.float32)
            # Compute the second convolution.
            second_result = scipy.signal.convolve2d(first_arr, second_kernel, mode='same')
            second_result = second_result - second_result.min()
            if second_result.max() != 0:
                second_result = second_result / second_result.max() * 255
            second_result = second_result.astype(np.uint8)
            temp_filename2 = f"temp_second_output_{name}.png"
            Image.fromarray(second_result).save(temp_filename2)
            second_conv_mobject = ImageMobject(temp_filename2)
            second_conv_mobject.scale_to_fit_height(2.5)
            # Position these to the left of the stacked outputs.
            second_conv_mobject.next_to(stacked_outputs, LEFT, buff=1)
            second_label = Text(f"Second {name}", font_size=24)
            second_label.next_to(second_conv_mobject, UP)

            # Animate the transformation from the first output to the second.
            self.play(Transform(first_out, second_conv_mobject), Write(second_label))
            self.wait(1)
            second_conv_outputs[name] = second_conv_mobject
            second_output_mobjects.append(second_conv_mobject)

        ##############################
        # Stage 5: Feed Final Convolutions into a Neural Network
        ##############################
        # Represent the neural network as a purple box.
        nn_box = RoundedRectangle(corner_radius=0.3, height=3, width=3, color=PURPLE)
        nn_label = Text("Neural Network", font_size=24, color=PURPLE)
        nn_group = VGroup(nn_box, nn_label)
        nn_group.to_edge(DOWN, buff=1)
        self.play(Create(nn_box), Write(nn_label))
        self.wait(0.5)

        # Draw arrows from each final convolved image into the neural network box.
        for conv_img in second_output_mobjects:
            arrow = Arrow(conv_img.get_right(), nn_box.get_left(), buff=0.1, color=WHITE)
            self.play(Create(arrow))
            self.wait(0.3)

        # Finally, display the neural network’s output.
        nn_output = MathTex("y", font_size=36, color=ORANGE)
        nn_output.next_to(nn_box, RIGHT, buff=0.5)
        self.play(Write(nn_output))
        self.wait(2)
