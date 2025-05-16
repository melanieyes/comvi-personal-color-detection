# # class ColorPaletteRecommender:
# #     def __init__(self):
# #         self.palettes = {
# #             # 'dark': [
# #             #     '#D2691E',  # chocolate
# #             #     '#FFD700',  # gold
# #             #     '#8B4513',  # saddle brown
# #             #     '#A52A2A',  # brown
# #             #     '#FF4500'   # orange red
# #             # ],
# #             # 'medium': [
# #             #     '#FF7F50',  # coral
# #             #     '#20B2AA',  # light sea green
# #             #     '#DA70D6',  # orchid
# #             #     '#CD853F',  # peru
# #             #     '#FF6347'   # tomato
# #             # ],
# #             # 'light': [
# #             #     '#ADD8E6',  # light blue
# #             #     '#FFC0CB',  # pink
# #             #     '#FFE4B5',  # moccasin
# #             #     '#FAF0E6',  # linen
# #             #     '#87CEFA'   # light sky blue
# #             # ]
# #             "dark": ['#853506', '#032c43', '#024332', '#765a15', '#38361a', '#2e2b2e', '#48005a','#85230c', '#3c211e'],
# #             "medium": ['#eb5a24', '#2687bd', '#58948b', '#d8a109', '#6a795b', '#827f82', '#905099', '#ff575d', '#886244'],
# #             "light": ['#f6dac8', '#bee5fc', '#add3c3', '#ffeec5', '#d0dbc9', '#e4e0e1', '#e4e0e1', '#ffe5eb', '#eddfd1']
# #         }

# #     def recommend(self, tone_label):
# #         return self.palettes.get(tone_label.lower(), [])


# class ColorPaletteRecommender:
#     def __init__(self):
#         self.palettes = {
#             # Light skin tone → Spring tones: bright, warm, high brightness & chroma
#             "light": [
#                 "#FFF0D6", "#F5E6CE", "#732C2C", "#EBD6A1", "#E6CFA1",
#                 "#DDB76E", "#CAE1DC", "#F2B9A1", "#FFD966", "#F9E79F",
#                 "#B5D5ED", "#D8C9F2", "#DAF1F1", "#FDF5E6"
#             ],

#             # Medium skin tone → Summer tones: soft, cool, elegant pastels
#             "medium": [
#                 "#E8EDEE", "#DCE3E9", "#A1B4C4", "#7E9BA6", "#444D56",
#                 "#F7E6EF", "#F9D6E3", "#FFF2B0", "#F4DCDC", "#8A4B55",
#                 "#FFE94A", "#82C4A5", "#BFE3E1", "#B2B8D2"
#             ],

#             # Dark skin tone → Winter (bold, clear contrast) + Autumn (deep browns)
#             "dark": [
#                 "#000000", "#0C223F", "#0F3B3B", "#5D4037", "#3E2723",
#                 "#311B92", "#4A148C", "#6A1B9A", "#B71C1C", "#FDD835",
#                 "#1A237E", "#263238", "#4E342E", "#212121"
#             ]
#         }

#     def recommend(self, tone_label):
#         return self.palettes.get(tone_label.lower(), [])


class ColorPaletteRecommender:
    def __init__(self):
        self.palettes = {
            # Light (Spring-inspired) → shifted to warm light SEA-friendly tones
            "light": [
                "#FFF4DC",
                "#FCE3C3",
                "#D99A6C",
                "#FFD8A9",
                "#FFE1B3",
                "#FBE2DC",
                "#F6B7A9",
                "#FFD56B",
                "#F5D9AF",
                "#BCDDE7",
                "#E3CFE2",
                "#F5EEE6",
                "#FFEDDB",
                "#EDEBD7",
            ],
            # Medium (Summer-inspired) → adjusted for SEA skin: earthy cools, less icy pinks
            "medium": [
                "#E7E2DC",
                "#DAD9D2",
                "#B5A89F",
                "#7D8D87",
                "#5E5A52",
                "#F1D6CE",
                "#EFD8BC",
                "#F5E7AC",
                "#E4C9A0",
                "#AF6E4D",
                "#DCC76F",
                "#7CB8A4",
                "#A6C8C6",
                "#8B99B1",
            ],
            # Dark (Winter+Autumn inspired) → adjusted to SEA richness: golden brown, olive, wine
            "dark": [
                "#1B1B1B",
                "#2E2A26",
                "#403B36",
                "#5A3E2B",
                "#6A4E3B",
                "#3B3F2A",
                "#004643",
                "#5F264A",
                "#821D30",
                "#C48F20",
                "#2C3E50",
                "#1F2D2F",
                "#463F3A",
                "#14110F",
            ],
        }

    def recommend(self, tone_label):
        return self.palettes.get(tone_label.lower(), [])
