Your task is to edit a md version of a published economics paper in markdown format to remove specific types of content.

- Remove any information that explicitly references the causal graph and its contents, including the causal graph itself
    - This is the only information you should remove from the paper
    - Only modify the text when it is necessary to remove the causal graph's information
    - Only remove explicit references to the causal graph's elements, such as variable names, feedback loops, arrow colors, a variable explicitly being included, etc. Do not remove other references and related information to the causal graph, such as discussing elements of the causal graph, its relationships generally, and similar information

- You can only edit the paper; do not attempt to edit the causal graph
    - The graph is supplied as a reference only in <causal_graph> tags
    - Do not attempt to edit anything before </causal_graph>; this is not part of the paper and will be rejected
    
You have access to a special tool called 'normalize' that can replace text. This is the only way you can modify the text. Be careful to ensure that the text you are replacing is only the causal graph's information, and that it exists verbatim in the text.

The normalize tool takes three parameters:
1. start_string: The beginning of the text to replace
2. end_string: The end of the text to replace
3. replacement: The text to insert instead

You can call normalize multiple times to make several targeted replacements in the document. All three parameters are required for each call.

- By default, normalize will locate the *first* occurrence of the start_string. As a workaround for when the same text appears verbatim multiple times, use a slightly longer start_string and include some of the original text in your replacement to maintain context.
- Do not "redact" the text; remove references entirely rather than replacing them with generic text.
- Both the start and end strings will be included in the text that gets replaced. Changes are applied in order, so ensure that any string you replace is not used in another replacement or an error will be thrown.

Respond only with JSON in the following format:
{
  "normalizations": [
    {"start": "text to find (beginning)", "end": "text to find (end)", "replacement": "text to insert instead"},
    ...
  ]
}