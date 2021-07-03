seperate_punctuation = [
    (r'([^\s])([!?,\.\(\):\\/\-"])', r'\1 \2'),
    (r'(\()(.)', r'\1 \2')
]

remove_mentions = [
    (r'@([^\s]){0,50}\s?', '')
]

remove_links = [
    (r'http(.{0,100})\s?', '')
]

concatenated_rules = [
    remove_links[0],
    seperate_punctuation[0],
    seperate_punctuation[1],
    remove_mentions[0]
]