# def map_mouth_shape(word: str, begin_time: float = None, end_time: float = None, lang: str = "en"):
#     """Return a mouth shape of a word"""
#     # Pyphen model
#     # Strip the word
#     word = word.strip()
#     # Get number of syllable
#     syllabels = pyphen.inserted(word)
#
#     # Get number of syllabel
#     list_syllabels = syllabels.split("-")
#
#     # Spacing
#     num_split = len(list_syllabels) + 1
#     # Get duration
#     if end_time < begin_time:
#         raise Exception("End time must be higher than begin time")
#
#     # Spacing
#     spacing = (end_time - begin_time) / num_split
#     # Define indexes
#     indexes = [*range(0, num_split)]
#
#     # Get time point respective with syllabel
#     time_points = []
#     for index in indexes:
#         start_point = index * spacing + begin_time
#         time_points.append(round(start_point, 2))
#
#     # Return mount shape
#     shapes = random.choices(mouth_shape_ref, k=len(time_points))
#     shapes[0] = "X"
#     return time_points, shapes