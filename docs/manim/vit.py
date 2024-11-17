from manim import *

FONT = "Helvetica Neue"


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

    text = Text("RCC", font=FONT)
    text.scale(0.75 * size / 3).next_to(rectangle.get_corner(UP + RIGHT), DOWN + LEFT, buff=0.1)

    # Group the shapes together
    return VGroup(rectangle, hemisphere, text)



class Title(Scene):
    def construct(self):
        text = Text("MiT-UB", color=RED, font_size=100)
        self.play(Write(text))
        self.wait(0.5)
        subtext = Text("Medical Imaging Transformer - UnBiased", font_size=36, t2c={"M": RED, "I": RED, "T": RED, "U": RED, "B": RED})
        subtext.next_to(text, DOWN)
        self.play(FadeIn(subtext))
        self.wait(2)


class Introduction(Scene):
    def construct(self):
        title = Text("Objectives", font_size=72)
        self.play(Write(title))
        self.wait(1)

        title.generate_target()
        title.target.to_edge(UL)
        title.target.scale(0.8)
        self.play(MoveToTarget(title))
        self.wait(0.5)

        objectives = [
            "Tailor transformer architecture for medical imaging",
            "Leverage self supervised learning to address data scarcity",
            "Accommodate a diverse range of medical imaging modalities",
            "Mitigate algorithmic bias"
        ]

        bullet_points = VGroup(
            *[Text(f"• {objective}", font_size=30) for objective in objectives]
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5)

        bullet_points.to_edge(LEFT)

        for bullet_point in bullet_points:
            self.play(Write(bullet_point))
            self.wait(1)
        self.wait(2)


class ViTInput(Scene):
    def construct(self):
        # Intro text
        text = Text("Processing of a Single Mammogram", font=FONT, color=WHITE)
        self.play(Write(text))
        self.wait(1.5)
        self.play(FadeOut(text))

        # Create the rectangle
        size = 3
        mammogram = draw_rcc(size=size)
        
        # Add the group to the scene
        self.play(DrawBorderThenFill(mammogram))
        self.wait(1)

        # Add an indicator to the first square using Brace
        brace_top = Brace(mammogram, direction=UP, buff=0.1)
        brace_left = Brace(mammogram, direction=LEFT, buff=0.1)
        indicator_text_top = Text("2304 px", font=FONT, color=WHITE)
        indicator_text_top.scale(0.5)
        indicator_text_left = Text("3072 px", font=FONT, color=WHITE)
        indicator_text_left.scale(0.5)
        indicator_text_top.next_to(brace_top, UP, buff=0.1)
        indicator_text_left.next_to(brace_left, LEFT, buff=0.1)
        braces = VGroup(brace_top, brace_left)
        indicator_text = VGroup(indicator_text_top, indicator_text_left)
        self.play(GrowFromEdge(braces, UL), Write(indicator_text))
        self.wait(2)

        # Add resize step text
        text = Text("Resize", font=FONT, color=RED, font_size=56)
        text.scale(0.5).next_to(mammogram, DOWN, buff=0.4)
        self.play(Write(text))
        self.wait(0.5)

        # Strike through the indicator text and update the size
        indicator_text_top_new = Text("384 px", font=FONT, color=RED)
        indicator_text_top_new.scale(0.5)
        indicator_text_top_new.next_to(brace_top, UP, buff=0.1)
        indicator_text_left_new = Text("512 px", font=FONT, color=RED)
        indicator_text_left_new.scale(0.5).next_to(brace_left, LEFT, buff=0.1)
        self.play(
            ReplacementTransform(indicator_text_top, indicator_text_top_new), 
            ReplacementTransform(indicator_text_left, indicator_text_left_new),
        )
        self.wait(1.5)

        # Fade out the brace and indicator text
        self.play(
            FadeOut(braces), 
            FadeOut(indicator_text_top_new), 
            FadeOut(indicator_text_left_new),
            FadeOut(text),
        )
        self.wait(1)

        # Show patching
        rows, cols = 4, 3
        patches = Table(
            [["x" for _ in range(cols)] for _ in range(rows)],
            v_buff=0.0,
            h_buff=0.0,
            include_outer_lines=False,
            line_config={"stroke_color": RED, "stroke_width": 1.5}
        )
        for row in patches.get_rows():
            row.set_opacity(0)
        patches.stretch_to_fit_width(mammogram.width)
        patches.stretch_to_fit_height(mammogram.height)
        text = Text("Patching", font=FONT, color=RED)
        text.scale(0.5).next_to(mammogram, DOWN, buff=0.4)
        self.play(DrawBorderThenFill(patches), Write(text))
        self.wait(0.5)

        # Add an indicator to the first square using Brace
        top_left = patches.get_cell((1, 1))
        brace_top = Brace(top_left, direction=UP, buff=0.1, color=RED)
        brace_left = Brace(top_left, direction=LEFT, buff=0.1, color=RED)
        indicator_text_top = Text("16px", font=FONT, color=RED)
        indicator_text_top.scale(0.5)
        indicator_text_top.next_to(brace_top, UP, buff=0.1)
        indicator_text_bottom = indicator_text_top.copy()
        indicator_text_bottom.next_to(brace_left, LEFT, buff=0.1)
        braces = VGroup(brace_top, brace_left)
        indicator_text = VGroup(indicator_text_top, indicator_text_bottom)
        self.play(GrowFromCenter(braces), Write(indicator_text))
        self.wait(2)

        # Remove the brace and indicator text from the scene
        self.play(FadeOut(braces), FadeOut(indicator_text), FadeOut(text))
        self.wait(1)

        # Create the arrows
        arrow_vert = DoubleArrow(start=mammogram.get_top(), end=mammogram.get_bottom(), buff=0.1, stroke_color=RED)
        arrow_vert.next_to(mammogram, LEFT, buff=0.6)
        arrow_horiz = DoubleArrow(start=mammogram.get_left(), end=mammogram.get_right(), buff=0.1, stroke_color=RED)
        arrow_horiz.next_to(mammogram, DOWN, buff=0.6)

        patches_text_vert = Text("32 patches", font=FONT, color=RED)
        patches_text_vert.scale(0.6)
        patches_text_vert.next_to(arrow_vert, LEFT, buff=0.1)
        patches_text_horiz = Text("24 patches", font=FONT, color=RED)
        patches_text_horiz.scale(0.6)
        patches_text_horiz.next_to(arrow_horiz, DOWN, buff=0.1)

        # Add the arrow and text to the scene
        token_text_group = VGroup(patches_text_vert, patches_text_horiz)
        self.play(GrowArrow(arrow_vert), GrowArrow(arrow_horiz), Write(token_text_group))
        self.wait(2)

        # Set up the equation
        eq = Text("32 x 24 = 768 tokens", font=FONT, color=RED)
        eq.scale(0.6)
        eq.next_to(mammogram, DOWN, buff=0.4)
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
        mammogram_with_patches = VGroup(mammogram, patches)
        mammogram_with_patches.generate_target()
        mammogram_with_patches.target.to_edge(LEFT, buff=2.0)

        # Add mammogram with patches to the scene
        self.play(
            MoveToTarget(mammogram_with_patches),
        )
        self.wait(1)

        # Create curved arrows from each patch to embeddings on the right
        linear_projection = Text("Linear Projection", font=FONT, color=RED)
        linear_projection.scale(0.6).next_to(mammogram_with_patches, DOWN, buff=0.4)
        self.play(Write(linear_projection), run_time=0.5)
        np.random.seed(0)
        embedding_length = 8
        ellipsis = Text("...", font=FONT, color=RED, font_size=72)
        embeddings = [
            Matrix([[x] for x in np.random.uniform(-9.9, 9.9, embedding_length).round(1)])
            for _ in range(4)
        ]
        # Stack embeddings side by side
        embedding_group = VGroup(*embeddings[:2], ellipsis, *embeddings[2:])
        embedding_group.arrange(RIGHT, buff=0.5)
        embedding_group.scale(0.4)  # Scale down to fit
        embedding_group.next_to(patches, RIGHT, buff=2.0)

        def animate_embedding(patch, embedding, align):
            # Create curved arrow
            start = patch.get_center()
            end = embedding.get_corner(align)
            arrow = CurvedArrow(
                start_point=start,
                end_point=end,
                color=RED,
                angle=-TAU/4 if align is UP else TAU/4
            )
            self.play(Create(arrow), run_time=0.3)
            self.play(FadeIn(embedding), run_time=0.2)
            self.wait(0.1)
            self.play(FadeOut(arrow), run_time=0.2)

        animate_embedding(patches.get_cell((1, 1)), embeddings[0], UP)
        animate_embedding(patches.get_cell((1, 2)), embeddings[1], UP)
        self.play(Write(ellipsis), FadeOut(linear_projection), run_time=0.5)
        animate_embedding(patches.get_cell((4, 2)), embeddings[2], DOWN)
        animate_embedding(patches.get_cell((4, 3)), embeddings[3], DOWN)
        visual_tokens = VGroup(*embeddings, ellipsis)
        visual_tokens_text = Text("Visual Tokens", font=FONT, color=RED)
        visual_tokens_text.scale(0.6).next_to(visual_tokens, DOWN, buff=0.4)
        self.play(Write(visual_tokens_text), run_time=0.5)
        self.wait(2)


        position_encoding = Text("Position Encoding", font=FONT, color=RED)
        position_encoding.scale(0.6).next_to(mammogram_with_patches, DOWN, buff=0.4)
        self.play(FadeOut(visual_tokens_text), Write(position_encoding), run_time=0.5)
        def animate_position(patch: Polygon, embedding: Matrix, align, color, coord):
            patch_target = patch.generate_target()
            patch_target.set_fill(color=color, opacity=0.8)
            embedding_target = embedding.generate_target()
            embedding_target.set_color(color)

            start = patch.get_center()
            end = embedding.get_corner(align)
            arrow = CurvedArrow(
                start_point=start,
                end_point=end,
                color=color,
                angle=-TAU/4 if align is UP else TAU/4
            )
            coord_text = Text(str(coord), font=FONT, color=color)
            coord_text.scale(0.5)
            coord_text.move_to(arrow.point_from_proportion(0.5) + np.array([0, 0.2, 0]))

            self.play(
                Create(arrow), 
                Create(coord_text),
                ReplacementTransform(patch, patch_target), 
                ReplacementTransform(embedding, embedding_target), 
                run_time=0.3,
            )
            self.wait(0.1)
            self.play(FadeOut(arrow), FadeOut(patch_target), FadeOut(coord_text), run_time=0.2)
            return embedding_target

        e1 = animate_position(patches.get_cell((1, 1)), embeddings[0], UP, RED, (1, 1))
        e2 = animate_position(patches.get_cell((1, 2)), embeddings[1], UP, ORANGE, (1, 2))
        self.play(FadeOut(position_encoding), run_time=0.25)
        e3 = animate_position(patches.get_cell((4, 2)), embeddings[2], DOWN, GREEN, (4, 2))
        e4 = animate_position(patches.get_cell((4, 3)), embeddings[3], DOWN, BLUE, (4, 3))
        embeddings = VGroup(e1, e2, e3, e4)
        self.play(FadeOut(patches), run_time=0.5)
        self.wait(1)

        tokens = VGroup(*embeddings, ellipsis)
        token_target = tokens.generate_target()
        token_target.move_to(ORIGIN)
        text = Text("Visual Tokens With Position Encoding", font=FONT, color=RED)
        text.scale(0.6).next_to(token_target, DOWN, buff=0.4)
        self.play(
            FadeOut(mammogram), 
            FadeOut(embeddings), 
            ReplacementTransform(tokens, token_target), 
            Write(text), 
            run_time=0.5,
        )
        self.wait(1)


class JEPA(Scene):
    def construct(self):
        # Draw the mammogram
        mammogram = draw_rcc(size=1.5)
        mammogram.to_edge(UL)
        text = Text("Mammogram", font=FONT, color=WHITE)
        text.scale(0.33).next_to(mammogram, DOWN, buff=0.1)
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
        text = Text("Teacher Network", font=FONT, color=WHITE)
        text.scale(0.33).next_to(teacher, DOWN, buff=0.1)
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
        text = Text("Teacher Embeddings", font=FONT, color=WHITE)
        text.scale(0.33).next_to(teacher_output, DOWN, buff=0.1)
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
        text = Text("Masked Mammogram", font=FONT, color=WHITE)
        text.scale(0.33).next_to(masked_mammogram, DOWN, buff=0.1)
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
        text = Text("Student Network", font=FONT, color=WHITE)
        text.scale(0.33).next_to(student, DOWN, buff=0.1)
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
        text = Text("Student Embeddings", font=FONT, color=WHITE)
        text.scale(0.33).next_to(unmasked_squares, DOWN, buff=0.1)
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
        text = Text("Predictor", font=FONT, color=WHITE)
        text.scale(0.33).next_to(predictor, DOWN, buff=0.1)
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
        text = Text("Predicted Embeddings", font=FONT, color=WHITE)
        text.scale(0.33).next_to(predictor_output, DOWN, buff=0.1)
        self.play(DrawBorderThenFill(arrow), DrawBorderThenFill(predictor_output), Write(text))
        self.wait(2)
            

class JEPAUpdate(Scene):

    def construct(self):
        # Draw the title
        title = Text("Backpropagation And Teacher Update", font=FONT, color=WHITE)
        title.scale(0.8).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(0.25)

        # Draw the targets and predictions
        target = Matrix([[0.1], [-3.0], [8.1], [-1.4], [7.2]]).set_color(RED)
        prediction = Matrix([[0.4], [-3.2], [8.9], [-1.0], [7.5]]).set_color(BLUE)
        target_text = Paragraph("Target\nEmbeddings", font=FONT, color=RED, alignment="center")
        predicted_text = Paragraph("Predicted\nEmbeddings", font=FONT, color=BLUE, alignment="center")
        target_text.scale(0.5)
        predicted_text.scale(0.5)
        spacing = 2.0
        target.move_to(ORIGIN + np.array([-spacing, 0, 0]))
        prediction.move_to(ORIGIN + np.array([spacing, 0, 0]))
        target_text.next_to(target, DOWN, buff=0.5)
        predicted_text.next_to(prediction, DOWN, buff=0.5)
        self.play(DrawBorderThenFill(target), Write(target_text), DrawBorderThenFill(prediction), Write(predicted_text))
        self.wait(2)

        target.generate_target()
        prediction.generate_target()
        target.target.to_edge(LEFT)
        prediction.target.to_edge(RIGHT)
        self.play(Unwrite(target_text), Unwrite(predicted_text), MoveToTarget(target), MoveToTarget(prediction))
        self.wait(2)
        

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
        predictor_update_text = Text("Predictor", font=FONT, color=WHITE)
        predictor_update_text.scale(0.5).next_to(predictor_update, DOWN, buff=0.5)
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
        student_update_text = Text("Student", font=FONT, color=WHITE)
        student_update_text.scale(0.5).next_to(student_update, DOWN, buff=0.5)
        self.play(DrawBorderThenFill(arrow), DrawBorderThenFill(student_update), Write(student_update_text))
        self.wait(2)

        # Draw the teacher update
        arrow = Arrow(
            start=student_update.get_corner(LEFT) + np.array([-0.5, 0, 0]), 
            end=student_update.get_corner(LEFT) + np.array([-1.5, 0, 0]), 
            buff=0.1, 
            stroke_color=RED,
        )
        arrow_text = Text("EMA Weights", font=FONT, color=RED)
        arrow_text.scale(0.25).next_to(arrow, DOWN, buff=0.1)
        teacher_update = create_neural_network()
        teacher_update.next_to(arrow, LEFT, buff=0.5)
        teacher_update_text = Text("Teacher", font=FONT, color=RED)
        teacher_update_text.scale(0.5).next_to(teacher_update, DOWN, buff=0.5)
        self.play(DrawBorderThenFill(arrow), DrawBorderThenFill(teacher_update), Write(teacher_update_text), Write(arrow_text))
        self.wait(2)

        update_text = MathTex(r"\theta_{i+1} = \theta_i \alpha + (1 - \alpha) \theta_i", color=RED)
        update_text.scale(0.75).to_edge(DOWN, buff=0.5)
        self.play(Write(update_text))
        self.wait(2)













        






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

        text = Text("RMLO", font=FONT)
        text.scale(0.75).next_to(rectangle.get_corner(UP + RIGHT), DOWN + LEFT, buff=0.1)

        # Group the shapes together
        mammogram = VGroup(rectangle, breast, text)
        
        # Add the group to the scene
        self.play(DrawBorderThenFill(mammogram))
        self.wait(2)

class Hemisphere3D(ThreeDScene):
    def construct(self):
        fast = False

        # Create the raw breast
        raw_breast = Surface(
            lambda u, v: np.array([
                np.cos(u) * np.sin(v),
                np.cos(v),
                np.sin(u) * np.sin(v),
            ]),
            u_range=[0, TAU],
            v_range=[0, PI/2],
            fill_color=DARK_GRAY,
            checkerboard_colors=[WHITE, WHITE],
            color=WHITE,
            fill_opacity=0.5,
            stroke_opacity=0.25,
        )

        # Create the dense tissue
        dense_tissue = Surface(
            lambda u, v: np.array([
                0.75 * np.cos(u) * np.sin(v),
                0.5 * np.cos(v),
                0.75 * np.sin(u) * np.sin(v),
            ]),
            u_range=[0, TAU],
            v_range=[0, PI/2],
            fill_color=DARK_GRAY,
            checkerboard_colors=[WHITE, WHITE],
            color=WHITE,
            fill_opacity=0.8,
            stroke_opacity=0.25,
        )
        dense_tissue.move_to(raw_breast.get_corner(DOWN), aligned_edge=DOWN)

        # Display the breast
        breast = VGroup(raw_breast, dense_tissue)
        axes = ThreeDAxes()
        axes.scale(0.5)
        x_label = Text("x").scale(0.5).next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("y").scale(0.5).next_to(axes.y_axis.get_end(), UP)
        z_label = Text("z").scale(0.5).rotate(PI/2, axis=RIGHT).rotate(PI, axis=OUT).next_to(axes.z_axis.get_end(), OUT)
        axes.add(x_label, y_label, z_label)
        self.set_camera_orientation(phi=PI/4, theta=PI/4)
        if fast:
            self.add(breast, axes)
        else:
            self.play(FadeIn(breast), DrawBorderThenFill(axes))
        self.wait(0.5)

        # Move the camera to point along y-axis
        phi = PI / 2
        theta = PI / 2
        gamma = 0

        if fast:
            self.set_camera_orientation(phi=phi, theta=theta, gamma=gamma)
            self.remove(axes)
        else:
            self.move_camera(phi=phi, theta=theta, run_time=1)
            self.play(FadeOut(axes))
        self.wait(2)

        # Draw rectangles aboev and below the breast
        paddle_size = (breast.width * 1.1, breast.height * 1.1, breast.depth * 0.1)
        top_paddle = Prism(dimensions=paddle_size, fill_color=BLUE, fill_opacity=0.5)
        top_paddle.move_to(breast.get_corner(OUT), aligned_edge=IN)
        bottom_paddle = Prism(dimensions=paddle_size, fill_color=BLUE, fill_opacity=0.5)
        bottom_paddle.move_to(breast.get_corner(IN), aligned_edge=OUT)
        if fast:
            self.add(top_paddle, bottom_paddle)
        else:
            self.play(FadeIn(top_paddle), FadeIn(bottom_paddle))
        self.wait(2)

        # Animate the breast compression
        compression_factor = 0.66
        clipping_factor = 0.8
        compressed_raw_breast = Surface(
            lambda u, v: np.array([
                np.cos(u) * np.sin(v),
                np.cos(v),
                compression_factor * (np.sin(u) * np.sin(v)).clip(-clipping_factor, clipping_factor),
            ]),
            u_range=[0, TAU],
            v_range=[0, PI/2],
            fill_color=DARK_GRAY,
            checkerboard_colors=[WHITE, WHITE],
            color=WHITE,
            fill_opacity=0.5,
            stroke_opacity=0.25,
        )
        compressed_dense_tissue = Surface(
            lambda u, v: np.array([
                0.75 * np.cos(u) * np.sin(v),
                0.5 * np.cos(v),
                compression_factor * (0.75 * np.sin(u) * np.sin(v)).clip(-clipping_factor, clipping_factor),
            ]),
            u_range=[0, TAU],
            v_range=[0, PI/2],
            fill_color=DARK_GRAY,
            checkerboard_colors=[WHITE, WHITE],
            color=WHITE,
            fill_opacity=0.8,
            stroke_opacity=0.25,
        )
        top_paddle.generate_target()
        top_paddle.target.move_to(compressed_raw_breast.get_corner(OUT), aligned_edge=IN)
        bottom_paddle.generate_target()
        bottom_paddle.target.move_to(compressed_raw_breast.get_corner(IN), aligned_edge=OUT)

        if fast:
            self.add(compressed_raw_breast, compressed_dense_tissue)
        else:
            self.play(
                ReplacementTransform(raw_breast, compressed_raw_breast), 
                ReplacementTransform(dense_tissue, compressed_dense_tissue),
                MoveToTarget(top_paddle),
                MoveToTarget(bottom_paddle),
            )
        self.wait(2)

        # Move back to 3d view
        phi, theta, gamma = PI / 4, PI / 4, 0
        if fast:
            self.remove(top_paddle, bottom_paddle)
            self.set_camera_orientation(phi=phi, theta=theta, gamma=gamma)
            self.add(axes)
        else:
            self.play(FadeIn(axes), FadeOut(top_paddle), FadeOut(bottom_paddle))
            self.move_camera(phi=phi, theta=theta, run_time=2)
        self.wait(2)

        # Draw the tomography projection locations
        N = 8
        angle = 30 * DEGREES
        angles = np.linspace(-angle + PI / 2, angle + PI / 2, N)
        arc_radius = 3
        lines = VGroup()
        for angle in angles:
            x = arc_radius * np.cos(angle)
            z = arc_radius * np.sin(angle)
            line = Line(start=[x, 0, z], end=[0, 0, 0], color=RED)
            lines.add(line)
        if fast:
            self.add(lines)
        else:
            for line in lines:
                self.play(DrawBorderThenFill(line), run_time=0.5)
        self.wait(2)





        #self.set_camera_orientation(phi=90 * DEGREES, theta=90 * DEGREES)
        ## Create the top rectangular prism
        #top_prism = Prism(dimensions=[1, 1, 0.33], fill_color=BLUE, fill_opacity=0.5)
        #top_prism.move_to(breast.get_corner(UP) + np.array([0, 0, 0.165]))

        ## Create the bottom rectangular prism
        #bottom_prism = Prism(dimensions=[1, 1, 0.33], fill_color=GREEN, fill_opacity=0.5)
        #bottom_prism.move_to(breast.get_corner(DOWN) + np.array([0, 0, -0.165]))

        # Add the prisms to the scene
        #self.add(top_prism, bottom_prism)
        
