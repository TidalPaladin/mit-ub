from manim import *

def create_neural_network(scale=1.0):
    radius = 0.1 * scale

    def create_node():
        result = Circle(radius=radius, color=WHITE)
        result.set_fill(WHITE, opacity=1.0)
        return result

    input_layer = VGroup(*[create_node() for _ in range(2)]).arrange(DOWN, buff=0.5)
    hidden_layer = VGroup(*[create_node() for _ in range(3)]).arrange(DOWN, buff=0.5)
    output_layer = VGroup(*[create_node() for _ in range(2)]).arrange(DOWN, buff=0.5)

    hidden_layer.next_to(input_layer, RIGHT, buff=0.5)
    output_layer.next_to(hidden_layer, RIGHT, buff=0.5)

    def connect_layers(layer1, layer2):
        lines = []
        for node1 in layer1:
            for node2 in layer2:
                line = Line(node1.get_center(), node2.get_center(), color=WHITE)
                lines.append(line)
        return VGroup(*lines)

    lines_in = connect_layers(input_layer, hidden_layer)
    lines_out = connect_layers(hidden_layer, output_layer)
    return VGroup(input_layer, hidden_layer, output_layer, lines_in, lines_out)


def draw_rcc(size=3):
    # Create the rectangle
    rectangle = Rectangle(
        width=size, 
        height=size*1.33, 
        fill_color=BLACK, 
        fill_opacity=1, 
        color=WHITE, 
        stroke_width=6,
    )
    
    # Create the hemisphere
    hemisphere = Arc(
        radius=size * 0.66, 
        angle=PI, 
        start_angle=3 * PI/2, 
        fill_color=DARK_GRAY, 
        fill_opacity=1, 
        color=WHITE, 
        stroke_width=6,
    )
    
    # Position the hemisphere to the left side of the rectangle
    hemisphere.move_to(rectangle.get_left() + np.array([hemisphere.radius / 2, 0.0, 0]))

    text = Text("RCC", font="Helvetica Neue")
    text.scale(0.75 * size / 3).next_to(rectangle.get_corner(UP + RIGHT), DOWN + LEFT, buff=0.1)

    # Group the shapes together
    return VGroup(rectangle, hemisphere, text)


class Title(Scene):
    def construct(self):
        text = Text("MiT-UB")
        self.play(Write(text))
        self.wait(1)
        subtext = Text("A self-supervised vision transformer for medical imaging")
        subtext.scale(0.5).next_to(text, DOWN)
        self.play(FadeIn(subtext))
        self.wait(2)


class JEPA(Scene):
    def construct(self):
        # Draw the mammogram
        mammogram = draw_rcc(size=1.5)
        mammogram.to_edge(UL)
        text = Text("Mammogram", font="Helvetica Neue", color=WHITE)
        text.scale(0.25).next_to(mammogram, DOWN, buff=0.1)
        self.play(DrawBorderThenFill(mammogram), Write(text))
        self.wait(2)

        # Draw the teacher network
        arrow = Arrow(
            start=mammogram.get_corner(RIGHT) + np.array([0.5, 0, 0]), 
            end=mammogram.get_corner(RIGHT) + np.array([1.5, 0, 0]), 
            buff=0.1, 
            stroke_color=WHITE,
        )
        teacher = create_neural_network()
        teacher.next_to(arrow, RIGHT, buff=0.5)
        text = Text("Teacher Network", font="Helvetica Neue", color=WHITE)
        text.scale(0.25).next_to(teacher, DOWN, buff=0.1)
        self.play(DrawBorderThenFill(arrow), DrawBorderThenFill(teacher), Write(text))
        self.wait(1)

        # Draw the teacher network output
        arrow = Arrow(
            start=teacher.get_corner(RIGHT) + np.array([0.5, 0, 0]), 
            end=teacher.get_corner(RIGHT) + np.array([1.5, 0, 0]), 
            buff=0.1, 
            stroke_color=WHITE,
        )
        square_size = mammogram.height / 4
        teacher_output = VGroup(*[
            Square(
                side_length=square_size,
                color=WHITE,
                stroke_width=2.0
            ).move_to(np.array([square_size * (0.5 + i), -square_size * (0.5 + j), 0]))
            for i in range(3) for j in range(4)
        ])
        teacher_output.next_to(arrow, RIGHT, buff=0.5)
        text = Text("Teacher Embeddings", font="Helvetica Neue", color=WHITE)
        text.scale(0.25).next_to(teacher_output, DOWN, buff=0.1)
        self.play(DrawBorderThenFill(arrow), DrawBorderThenFill(teacher_output), Write(text))
        self.wait(2)

        # Draw the masked mammogram
        masked_mammogram = mammogram.copy()
        masked_mammogram.to_edge(DL)
        all_squares = [
            Square(
                side_length=square_size,
                color=WHITE,
                fill_color=BLACK,
                fill_opacity=1.0,
                stroke_width=2.0,
            ).move_to(
                masked_mammogram.get_corner(UL) + np.array([square_size * i, square_size * -j, 0]),
                aligned_edge=UL,
            )
            for i in range(3) for j in range(4)
        ]
        mask_ratio = 0.5
        np.random.seed(0)
        sampled_mask = np.random.choice(len(all_squares), size=int(mask_ratio * len(all_squares)), replace=False)
        sampled_mask = [all_squares[i] for i in sampled_mask]
        unmasked = VGroup(*[m for m in all_squares if m not in sampled_mask])
        mask = VGroup(*sampled_mask)
        for m in unmasked:
            m.set_fill(BLUE, opacity=0.5)
        masked_mammogram = VGroup(masked_mammogram, mask, unmasked)
        text = Text("Masked Mammogram", font="Helvetica Neue", color=WHITE)
        text.scale(0.25).next_to(masked_mammogram, DOWN, buff=0.1)
        self.play(DrawBorderThenFill(masked_mammogram), Write(text))
        self.wait(2)

        # Draw the student network
        arrow = Arrow(
            start=masked_mammogram.get_corner(RIGHT) + np.array([0.5, 0, 0]), 
            end=masked_mammogram.get_corner(RIGHT) + np.array([1.5, 0, 0]), 
            buff=0.1, 
            stroke_color=WHITE,
        )   
        student = create_neural_network()
        student.next_to(arrow, RIGHT, buff=0.5)
        text = Text("Student Network", font="Helvetica Neue", color=WHITE)
        text.scale(0.25).next_to(student, DOWN, buff=0.1)
        self.play(DrawBorderThenFill(arrow), DrawBorderThenFill(student), Write(text))
        self.wait(1)

        # Draw the student network output
        arrow = Arrow(
            start=student.get_corner(RIGHT) + np.array([0.5, 0, 0]), 
            end=student.get_corner(RIGHT) + np.array([1.5, 0, 0]), 
            buff=0.1, 
            stroke_color=WHITE,
        )
        unmasked_squares = VGroup(*[square.copy() for square in unmasked])
        for s in unmasked_squares:
            s.set_fill(BLUE, opacity=1.0)
        unmasked_squares.next_to(arrow, RIGHT, buff=0.5)
        text = Text("Student Embeddings", font="Helvetica Neue", color=WHITE)
        text.scale(0.25).next_to(unmasked_squares, DOWN, buff=0.1)
        self.play(DrawBorderThenFill(arrow), DrawBorderThenFill(unmasked_squares), Write(text))
        self.wait(1)

        # Draw the queries
        query_ratio = 0.25
        np.random.seed(1)
        sampled_query_mask = np.random.choice(len(all_squares), size=int(query_ratio * len(all_squares)), replace=False)
        query_mask = VGroup(*[all_squares[i].copy() for i in sampled_query_mask])
        for q in query_mask:
            q.set_fill(RED, opacity=1.0)
        query_mask.move_to(teacher_output.get_corner(UL), aligned_edge=UL)
        self.play(DrawBorderThenFill(query_mask))
        self.wait(1)

        # Draw the predictor
        predictor = create_neural_network()
        arrow = Arrow(
            start=unmasked_squares.get_corner(RIGHT) + np.array([0.5, 0, 0]), 
            end=unmasked_squares.get_corner(RIGHT) + np.array([1.5, 0, 0]), 
            buff=0.1, 
            stroke_color=WHITE,
        )
        predictor.next_to(arrow, RIGHT, buff=0.5)
        text = Text("Predictor", font="Helvetica Neue", color=WHITE)
        text.scale(0.25).next_to(predictor, DOWN, buff=0.1)
        self.play(DrawBorderThenFill(arrow), DrawBorderThenFill(predictor), Write(text))
        self.wait(1)

        # Draw the predictor output
        arrow = Arrow(
            start=predictor.get_corner(UP) + np.array([0, 0.5, 0]), 
            end=np.array([predictor.get_corner(UP)[0], teacher_output.get_corner(DOWN)[1] - 0.5, 0]), 
            buff=0.1, 
            stroke_color=WHITE,
        )
        predictor_output = VGroup(*[s.copy() for s in query_mask])
        for s in predictor_output:
            s.set_fill(GREEN, opacity=1.0)
        predictor_output.move_to(
            np.array([predictor.get_corner(UP)[0], teacher_output.get_corner(RIGHT)[1], 0]),
        )
        text = Text("Predicted Embeddings", font="Helvetica Neue", color=WHITE)
        text.scale(0.25).next_to(predictor_output, DOWN, buff=0.1)
        self.play(DrawBorderThenFill(arrow), DrawBorderThenFill(predictor_output), Write(text))
        self.wait(2)
            

class JEPAUpdate(Scene):

    def construct(self):
        # Draw the title
        title = Text("Backpropagation And Teacher Update", font="Helvetica Neue", color=WHITE)
        title.scale(0.8).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(0.25)

        # Draw the targets and predictions
        square_size = 1.0
        target = Square(
            side_length=square_size,
            color=WHITE,
            fill_color=RED,
            fill_opacity=1.0,
            stroke_width=2.0
        )
        target_text = Text("Target Embeddings", font="Helvetica Neue", color=RED)
        target_text.scale(0.25)
        prediction = Square(
            side_length=square_size,
            color=WHITE,
            fill_color=BLUE,
            fill_opacity=1.0,
            stroke_width=2.0
        )
        predicted_text = Text("Predicted Embeddings", font="Helvetica Neue", color=BLUE)
        predicted_text.scale(0.25)
        target.move_to(ORIGIN + np.array([-square_size, 0, 0]))
        prediction.move_to(ORIGIN + np.array([square_size, 0, 0]))
        target_text.next_to(target, DOWN, buff=0.1)
        predicted_text.next_to(prediction, DOWN, buff=0.1)
        self.play(DrawBorderThenFill(target), Write(target_text), DrawBorderThenFill(prediction), Write(predicted_text))
        self.wait(1)

        # Define loss function
        brace = Brace(VGroup(target, prediction), direction=UP, buff=0.2)
        brace_text = MathTex(r"\mathcal{L}(x, \hat{x}) := \text{SmoothL1Loss}(x, \hat{x})", color=WHITE)
        brace_text.scale(0.75)
        brace_text.next_to(brace, UP, buff=0.1)
        self.play(GrowFromCenter(brace), Write(brace_text))
        self.wait(2)

        # Morph the loss text into a standalone symbol and hide everything else
        loss_symbol = MathTex(r"\mathcal{L}(x, \hat{x})", color=WHITE)
        loss_symbol.to_edge(RIGHT)
        self.play(Transform(brace_text, loss_symbol), FadeOut(brace), FadeOut(target), FadeOut(prediction), FadeOut(target_text), FadeOut(predicted_text))
        self.wait(1)

        # Draw the predictor update
        arrow = Arrow(
            start=loss_symbol.get_corner(LEFT) + np.array([-0.5, 0, 0]), 
            end=loss_symbol.get_corner(LEFT) + np.array([-1.5, 0, 0]), 
            buff=0.1, 
            stroke_color=WHITE,
        )
        predictor_update = create_neural_network()
        predictor_update.next_to(arrow, LEFT, buff=0.5)
        predictor_update_text = Text("Predictor", font="Helvetica Neue", color=WHITE)
        predictor_update_text.scale(0.25).next_to(predictor_update, DOWN, buff=0.5)
        self.play(DrawBorderThenFill(arrow), DrawBorderThenFill(predictor_update), Write(predictor_update_text))
        self.wait(2)

        # Draw the student update
        arrow = Arrow(
            start=predictor_update.get_corner(LEFT) + np.array([-0.5, 0, 0]), 
            end=predictor_update.get_corner(LEFT) + np.array([-1.5, 0, 0]), 
            buff=0.1, 
            stroke_color=WHITE,
        )
        student_update = create_neural_network()
        student_update.next_to(arrow, LEFT, buff=0.5)
        student_update_text = Text("Student", font="Helvetica Neue", color=WHITE)
        student_update_text.scale(0.25).next_to(student_update, DOWN, buff=0.5)
        self.play(DrawBorderThenFill(arrow), DrawBorderThenFill(student_update), Write(student_update_text))
        self.wait(2)

        # Draw the teacher update
        arrow = Arrow(
            start=student_update.get_corner(LEFT) + np.array([-0.5, 0, 0]), 
            end=student_update.get_corner(LEFT) + np.array([-1.5, 0, 0]), 
            buff=0.1, 
            stroke_color=RED,
        )
        arrow_text = Text("EMA Weights", font="Helvetica Neue", color=RED)
        arrow_text.scale(0.25).next_to(arrow, DOWN, buff=0.1)
        teacher_update = create_neural_network()
        teacher_update.next_to(arrow, LEFT, buff=0.5)
        teacher_update_text = Text("Teacher", font="Helvetica Neue", color=RED)
        teacher_update_text.scale(0.25).next_to(teacher_update, DOWN, buff=0.5)
        self.play(DrawBorderThenFill(arrow), DrawBorderThenFill(teacher_update), Write(teacher_update_text), Write(arrow_text))
        self.wait(2)

        update_text = MathTex(r"\theta_{i+1} = \theta_i \alpha + (1 - \alpha) \theta_i", color=RED)
        update_text.scale(0.75).to_edge(DOWN, buff=0.5)
        self.play(Write(update_text))
        self.wait(2)













        


class Mammo2(Scene):
    def construct(self):
        # Intro text
        text = Text("Processing of a Single Mammogram", font="Helvetica Neue", color=WHITE)
        self.play(Write(text))
        self.wait(1.5)
        self.play(FadeOut(text))

        # Create the rectangle
        size = 3
        rectangle = RoundedRectangle(
            width=size, 
            height=size*1.33, 
            fill_color=BLACK, 
            fill_opacity=1, 
            color=WHITE, 
            stroke_width=6,
            corner_radius=0.1
        )
        
        # Create the hemisphere
        hemisphere = Arc(
            radius=size * 0.66, 
            angle=PI, 
            start_angle=3 * PI/2, 
            fill_color=DARK_GRAY, 
            fill_opacity=1, 
            color=WHITE, 
            stroke_width=6,
        )
        
        # Position the hemisphere to the left side of the rectangle
        hemisphere.move_to(rectangle.get_left() + np.array([hemisphere.radius / 2, 0.0, 0]))

        text = Text("RCC", font="Helvetica Neue")
        text.scale(0.75).next_to(rectangle.get_corner(UP + RIGHT), DOWN + LEFT, buff=0.1)

        # Group the shapes together
        mammogram = VGroup(rectangle, hemisphere, text)
        
        # Add the group to the scene
        self.play(DrawBorderThenFill(mammogram))
        self.wait(2)

        # Add an indicator to the first square using Brace
        brace_top = Brace(rectangle, direction=UP, buff=0.1)
        brace_left = Brace(rectangle, direction=LEFT, buff=0.1)
        indicator_text_top = Text("2304 px", font="Helvetica Neue", color=WHITE)
        indicator_text_top.scale(0.5)
        indicator_text_left = Text("3072 px", font="Helvetica Neue", color=WHITE)
        indicator_text_left.scale(0.5)
        indicator_text_top.next_to(brace_top, UP, buff=0.1)
        indicator_text_left.next_to(brace_left, LEFT, buff=0.1)

        # Add the brace and indicator text to the scene
        braces = VGroup(brace_top, brace_left)
        indicator_text = VGroup(indicator_text_top, indicator_text_left)
        self.play(GrowFromEdge(braces, UL), Write(indicator_text))
        self.wait(2)

        # Fade out the brace and indicator text
        self.play(FadeOut(braces), FadeOut(indicator_text))
        self.wait(1)

        # Define the number of rows and columns for the grid
        rows, cols = 4, 3
        square_size = size / cols

        # Create the grid of squares
        squares = VGroup()
        for i in range(rows):
            for j in range(cols):
                square = Square(
                    side_length=square_size,
                    stroke_color=RED,
                    stroke_width=1,
                    fill_opacity=0
                )
                square.move_to(rectangle.get_corner(UP + LEFT) + np.array([(j + 0.5) * square_size, -(i + 0.5) * square_size, 0]))
                squares.add(square)

        # Add text below the rectangle
        text = Text("Decompose image into non-overlapping patches", font="Helvetica Neue", color=RED)
        text.scale(0.5).next_to(rectangle, DOWN, buff=0.4)

        # Add the grid of squares to the scene
        self.play(DrawBorderThenFill(squares), Write(text))
        self.wait(0.5)

        # Add an indicator to the first square using Brace
        brace_top = Brace(squares[0], direction=UP, buff=0.1)
        brace_left = Brace(squares[0], direction=LEFT, buff=0.1)
        indicator_text_top = Text("16px", font="Helvetica Neue", color=WHITE)
        indicator_text_top.scale(0.5)
        indicator_text_top.next_to(brace_top, UP, buff=0.1)
        indicator_text_bottom = indicator_text_top.copy()
        indicator_text_bottom.next_to(brace_left, LEFT, buff=0.1)

        # Add the brace and indicator text to the scene
        braces = VGroup(brace_top, brace_left)
        indicator_text = VGroup(indicator_text_top, indicator_text_bottom)
        self.play(GrowFromCenter(braces), Write(indicator_text))
        self.wait(2)

        # Remove the brace and indicator text from the scene
        self.play(FadeOut(braces), FadeOut(indicator_text), FadeOut(text))
        self.wait(1)

        # Create the arrows
        arrow_vert = DoubleArrow(start=rectangle.get_top(), end=rectangle.get_bottom(), buff=0.1, stroke_color=WHITE)
        arrow_vert.next_to(rectangle, LEFT, buff=0.5)
        arrow_horiz = DoubleArrow(start=rectangle.get_left(), end=rectangle.get_right(), buff=0.1, stroke_color=WHITE)
        arrow_horiz.next_to(rectangle, DOWN, buff=0.5)

        patches_text_vert = Text("192 patches", font="Helvetica Neue", color=WHITE)
        patches_text_vert.scale(0.5)
        patches_text_vert.next_to(arrow_vert, LEFT, buff=0.1)
        patches_text_horiz = Text("144 patches", font="Helvetica Neue", color=WHITE)
        patches_text_horiz.scale(0.5)
        patches_text_horiz.next_to(arrow_horiz, DOWN, buff=0.1)

        # Add the arrow and text to the scene
        token_text_group = VGroup(patches_text_vert, patches_text_horiz)
        self.play(GrowArrow(arrow_vert), GrowArrow(arrow_horiz), Write(token_text_group))
        self.wait(2)

        # Set up the equation
        eq = Text("192 x 144 = 27,648 tokens", font="Helvetica Neue", color=WHITE)
        eq.scale(0.5)
        eq.next_to(rectangle, DOWN, buff=0.4)
        group = VGroup(patches_text_vert, patches_text_horiz)
        self.play(
            FadeOut(arrow_vert), 
            FadeOut(arrow_horiz), 
            ReplacementTransform(group, eq),
        )
        self.wait(2)

        # Fade out the equation
        self.play(FadeOut(eq))
        self.wait(1)

        # Move the mammogram with patches to the left of the frame
        mammogram_with_patches = VGroup(mammogram, squares)
        mammogram_with_patches.generate_target()
        mammogram_with_patches.target.to_edge(LEFT, buff=2.0)

        # Add mammogram with patches to the scene
        self.play(
            MoveToTarget(mammogram_with_patches),
        )
        self.wait(2)

        # Loop over each of the squares
        linear_projection = Text("Linear Projection", font="Helvetica Neue", color=RED)
        linear_projection.scale(0.5).next_to(mammogram_with_patches, DOWN, buff=0.4)
        self.play(Write(linear_projection))
        embeddings = VGroup()
        for square in squares:
            # Fill the square with a blue background
            square.set_fill(BLACK, opacity=1.0)
            square.set_stroke(WHITE, width=2)
            
            # Generate a random number between -10 and 10
            random_number = np.random.uniform(-10, 10)
            
            # Create a text object with the random number, formatted to one decimal place
            number_text = Text(f"{random_number:.1f}", font="Helvetica Neue", color=WHITE)
            number_text.scale(0.5).move_to(square.get_center())
            embeddings.add(VGroup(square, number_text))
            
            # Add the number text to the scene
            self.play(FadeIn(square), Write(number_text), run_time=0.2)
            self.wait(0.05)
        self.remove(mammogram)
        self.play(FadeOut(linear_projection))
        self.wait(2)

        # Move the squares to the center of the frame
        for i, square in enumerate(embeddings):
            square.generate_target()
            offset = i * square.width - len(embeddings) * square.width / 2 + square.width / 2
            scale = 0.75
            square.target.scale(scale)
            offset *= scale
            square.target.move_to(ORIGIN + np.array([offset, 0, 0]))
            self.play(MoveToTarget(square), run_time=0.1)

        # Add the label text
        text = Text("Visual Tokens", font="Helvetica Neue", color=RED)
        text.scale(0.5).next_to(embeddings, DOWN, buff=0.4)
        self.play(Write(text))
        self.wait(1)




class CaseScore(Scene):
    def construct(self):
        svg = SVGMobject("svgs/lmlo.svg")

        # Filter elements by their ID or class name
        # Assume 'group_name' is the ID or class of the group you want to display
        self.play(DrawBorderThenFill(svg))
        self.wait(2)


class MLO(Scene):
    def construct(self):
        # Create the rectangle
        size = 3
        rectangle = RoundedRectangle(
            width=size, 
            height=size*1.33, 
            fill_color=BLACK, 
            fill_opacity=1, 
            color=WHITE, 
            stroke_width=6,
            corner_radius=0.1
        )
        
        # Create the hemisphere
        breast = CubicBezier(
            rectangle.get_corner(UP + LEFT) + np.array([0.1 * size, 0, 0]),
            rectangle.get_left() + np.array([size * 0.66, -size * 0.25, 0]),
            rectangle.get_left() + np.array([size * 0.66, -size * 0.66, 0]),
            rectangle.get_corner(DOWN + LEFT),
        )
        breast.set_fill(DARK_GRAY, opacity=1)
        breast.set_stroke(WHITE, width=6)
        
        ## Position the hemisphere to the left side of the rectangle
        #breast.move_to(rectangle.get_left() + np.array([breast.radius / 2, 0.0, 0]))

        text = Text("RMLO", font="Helvetica Neue")
        text.scale(0.75).next_to(rectangle.get_corner(UP + RIGHT), DOWN + LEFT, buff=0.1)

        # Group the shapes together
        mammogram = VGroup(rectangle, breast, text)
        
        # Add the group to the scene
        self.play(DrawBorderThenFill(mammogram))
        self.wait(2)