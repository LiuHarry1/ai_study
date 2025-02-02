from difflib import HtmlDiff

def generate_html_diff(text1, text2):
    # Split the texts into lines
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()

    # Create an HtmlDiff instance
    differ = HtmlDiff()

    # Generate the HTML diff
    html_diff = differ.make_file(lines1, lines2, fromdesc='Original Text', todesc='Modified Text')

    return html_diff


# Example input texts
text1 = "quick brown fox jumps here there over the lazy dog."
text2 = "The quick fox leaps over a lazy cat."

# Generate the HTML diff
html_result = generate_html_diff(text1, text2)

# Save the HTML to a file for viewing
with open("text_diff.html", "w") as f:
    f.write(html_result)

print("HTML diff generated and saved to 'text_diff.html'")
