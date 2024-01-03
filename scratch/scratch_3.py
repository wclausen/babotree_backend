import markdown
from bs4 import BeautifulSoup

def find_section_boundaries(md_text):
    # Convert Markdown to HTML
    html = markdown.markdown(md_text)

    # Parse HTML
    soup = BeautifulSoup(html, 'html.parser')

    # Find all headers (h1, h2, h3, etc.)
    headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    print(headers)

    sections = []
    for i, header in enumerate(headers):
        # Determine the level of the current header
        level = int(header.name[1])

        # Find the next header of the same or higher level
        next_header = None
        for next_elem in header.find_all_next(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            next_header = next_elem
            break

        # Get all content between current header and next header
        content = []
        for elem in header.next_siblings:
            if elem == next_header:
                break
            content.append(str(elem))

        sections.append((header, ''.join(content)))

    return sections

# Example usage
md_text = """
The Overstory: Key Concepts
=========================

I. The interconnectedness of life
---------------------------------

* Trees communicate and support each other through chemical signals, fungal networks, and shared resources (aboveground and belowground)
* Human perception of trees and nature is limited and often inaccurate

II. The importance of diversity and adaptation
---------------------------------------------

* There are many ways to branch, grow, and adapt to changing conditions
* Every piece of earth needs a new way to grip it
* Evolution and adaptation are ongoing processes

III. The impact of human activity on the natural world
------------------------------------------------------

* Human activities such as logging, agriculture, and urbanization have significant impacts on forests and trees
* The law often fails to protect the natural world from harm
* The need for justice is like a growing hunger that must be addressed

IV. Memory and legacy
--------------------

* Memory is a way for life to talk to the future
* People and trees leave legacies that can inspire and guide future generations

V. The power of stories and narratives
--------------------------------------

* Stories and narratives have the power to change people's minds and inspire action
* The best arguments in the world won't change a person's mind, but a good story might

VI. The limitations of human perception and understanding
-------------------------------------------------------

* Human perception is limited and often inaccurate
* Consciousness itself is a flavor of madness, set against the thoughts of the green world
* People see likeness as the sole problem of men

VII. The importance of imagination and creativity
-------------------------------------------------

* Imagination and creativity are unique to life and have the power to bridge past and future, earth and sky
* The ability to see, all at once, in all its concurrent branches, all its many hypotheticals, is a mystery that inspires awe and wonder.
"""

sections = find_section_boundaries(md_text)
for header, content in sections:
    print(f"Section starts with: {header.text}\nContent: {content.strip()}")
    print("-----")