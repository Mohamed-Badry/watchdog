import glob, re

for f in glob.glob('src/lib/components/charts/*.svelte'):
    if 'ResponsivePlot' in f or 'SparklinePlot' in f: continue
    
    with open(f, 'r') as file:
        content = file.read()
    
    if '<Plot' not in content:
        continue

    # Handle imports:
    # Match `import { Plot, ... }` or `import { ..., Plot, ... }` or `import { ..., Plot }`
    # It's easier to just do text replacement
    content = re.sub(r'import\s+\{\s*Plot\s*,\s*', 'import { ', content)
    content = re.sub(r',\s*Plot\s*\}', ' }', content)
    content = re.sub(r'import\s+\{\s*Plot\s*\}\s*from\s*[\'"]svelteplot[\'"];?', '', content)
    
    if 'ResponsivePlot' not in content:
        content = content.replace("<script lang=\"ts\">", "<script lang=\"ts\">\n  import ResponsivePlot from './ResponsivePlot.svelte';")
        
    content = content.replace('<Plot', '<ResponsivePlot')
    content = content.replace('</Plot>', '</ResponsivePlot>')
    
    with open(f, 'w') as file:
        file.write(content)

print("Done")
