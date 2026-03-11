from manim import (
    DOWN,
    UP,
    Create,
    FadeOut,
    MathTex,
    ReplacementTransform,
    Scene,
    SurroundingRectangle,
    Text,
    TexTemplate,
    Title,
    TransformMatchingTex,
    VGroup,
    Write,
)

diffcoeff_template = TexTemplate().add_to_preamble(r"\usepackage{diffcoeff}")


# class MovingFrameBox(Scene):
#     def construct(self):
#         text = MathTex(
#             r"\diff{}{x}f(x)g(x)=",
#             r"f(x)\diff{g(x)}{x}",
#             "+",
#             r"g(x)\diff{f(x)}{x}",
#             tex_template=diffcoeff_template,
#         )
#         self.play(Write(text))
#         framebox1 = SurroundingRectangle(text[1], buff=0.1)
#         framebox2 = SurroundingRectangle(text[3], buff=0.1)
#         self.play(
#             Create(framebox1),
#         )
#         self.wait()
#         self.play(
#             ReplacementTransform(framebox1, framebox2),
#         )
#         self.wait()


class LaneEmdenDerivation(Scene):
    def construct(self):
        # Part 1: Abstract Equations
        title_abstract = Title(
            "Equations of Stellar Structure",
            match_underline_width_to_text=True,
        ).scale(0.8)  # ⋆⭒˚.⋆🔭 🚀🌙🪐🌌⭐☄
        self.play(Write(title_abstract))

        lane_emden_eq = MathTex(
            r"\frac{1}{\xi^2}\diff*{\left(\xi^2\diff{\theta}{\xi}\right)}{\xi}+\theta^n=0.",
            tex_template=diffcoeff_template,
        ).scale(0.8)
        lane_emden_text = Text("Lane-Emden Equation", font_size=20).next_to(
            lane_emden_eq, DOWN, buff=0.4
        )

        chandrasekhar_eq = MathTex(
            r"\frac{1}{\xi^2}\diff*{\left(\xi^2\diff{\theta}{\xi}\right)}{\xi}-e^{-\theta}=0.",
            tex_template=diffcoeff_template,
        ).scale(0.8)
        chandrasekhar_text = Text(
            "Emden-Chandrasekhar Equation (Isothermal case)", font_size=20
        ).next_to(chandrasekhar_eq, DOWN, buff=0.4)

        white_dwarf_eq = MathTex(
            r"\frac{1}{\xi^2}\diff*{\left(\xi^2\diff{\theta}{\xi}\right)}{\xi}+(\theta^2-C)^{\frac{3}{2}}=0.",
            tex_template=diffcoeff_template,
        ).scale(0.8)
        white_dwarf_text = Text(
            "Chandrasekhar's white dwarf equation", font_size=20
        ).next_to(white_dwarf_eq, DOWN, buff=0.4)

        VGroup(
            VGroup(lane_emden_eq, lane_emden_text),
            VGroup(white_dwarf_eq, white_dwarf_text),
            VGroup(chandrasekhar_eq, chandrasekhar_text),
        ).arrange(DOWN, buff=0.22).center()

        self.play(Write(lane_emden_eq), Write(lane_emden_text))
        self.wait(2)
        self.play(Write(white_dwarf_eq), Write(white_dwarf_text))
        self.wait(2)
        self.play(Write(chandrasekhar_eq), Write(chandrasekhar_text))
        self.wait(3)
        self.play(
            FadeOut(title_abstract),
            FadeOut(lane_emden_eq),
            FadeOut(lane_emden_text),
            FadeOut(chandrasekhar_eq),
            FadeOut(chandrasekhar_text),
            FadeOut(white_dwarf_eq),
            FadeOut(white_dwarf_text),
        )
        self.wait()

        # Part 2: Derivation
        title_derivation = Title(
            "Derivation of Lane-Emden Equation", match_underline_width_to_text=True
        ).scale(0.8)
        self.play(Write(title_derivation))

        # Equations
        eq1_tex = r"\diff{P\left(r\right)}{r}=-\frac{Gm\left(r\right)}{r^2}\rho\left(r\right).\quad (1)"
        eq2_tex = r"\diff{m\left(r\right)}{r}=4\pi r^2\rho\left(r\right).\quad (2)"
        eq1 = (
            MathTex(eq1_tex, tex_template=diffcoeff_template)
            .to_edge(UP, buff=1.2)
            .scale(0.8)
        )
        eq2 = (
            MathTex(eq2_tex, tex_template=diffcoeff_template)
            .next_to(eq1, DOWN, buff=0.7)
            .scale(0.8)
        )

        self.play(Write(eq1))
        self.play(Write(eq2))
        self.wait(2)

        # Step 1: Differentiate (1)
        step1_text = MathTex(
            r"\text{Multiply (1) by }\frac{r^2}{\rho\left(r\right)}\text{ and differentiate w.r.t } r",
            font_size=28,
        ).next_to(eq2, DOWN, buff=1)
        self.play(Write(step1_text))
        self.wait(1)

        eq3_tex = r"\diff*{\left(\frac{r^2}{\rho\left(r\right)}\diff{P\left(r\right)}{r}\right)}{r}=-G\diff{m\left(r\right)}{r}.\quad (3)"
        eq3 = (
            MathTex(eq3_tex, tex_template=diffcoeff_template)
            .next_to(step1_text, DOWN, buff=0.5)
            .scale(0.8)
        )
        self.play(TransformMatchingTex(eq1.copy(), eq3, transform_mismatches=True))
        self.wait(2)

        # Step 2: Substitute (2) into (3)
        step2_text = Text("Substitute (2) into (3)", font_size=20).move_to(step1_text)
        self.play(ReplacementTransform(step1_text, step2_text))

        eq4_tex = r"\frac{1}{r^2}\diff*{\left(\frac{r^2}{\rho\left(r\right)}\diff{P\left(r\right)}{r}\right)}{r}=-4\pi G\rho\left(r\right)."
        eq4 = MathTex(eq4_tex, tex_template=diffcoeff_template).move_to(eq3).scale(0.8)
        self.play(
            TransformMatchingTex(
                VGroup(eq3, eq2.copy()), eq4, transform_mismatches=True
            )
        )
        self.wait(2)

        # Step 3: Rearrange
        step3_text = Text(
            "Rearranging gives the Lane-Emden Equation", font_size=20
        ).move_to(step2_text)
        self.play(ReplacementTransform(step2_text, step3_text))

        final_eq_tex = r"\frac{1}{r^2}\diff*{\left(\frac{r^2}{\rho\left(r\right)}\diff{P\left(r\right)}{r}\right)}{r}=-4\pi G\rho\left(r\right).\quad (4)"
        final_eq = (
            MathTex(final_eq_tex, tex_template=diffcoeff_template)
            .move_to(eq4)
            .scale(0.8)
        )
        self.play(ReplacementTransform(eq4, final_eq))
        self.wait(2)

        # Final Scene
        self.play(
            FadeOut(title_derivation),
            FadeOut(eq1),
            FadeOut(eq2),
            FadeOut(step3_text),
        )
        self.play(final_eq.animate.center().scale(1.2))
        self.wait(4)
