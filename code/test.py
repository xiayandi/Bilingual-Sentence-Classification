"""
this module is for test only
"""

vocablst = set()
with open('../data/eng_dict_desc.lst', 'r') as reader:
    datalines = reader.readlines()
for line in datalines:
    if line.strip():
        items = line.split()
        if len(items)<2:
            continue
        if items[0] == 'Usage':
            continue
        if items[0][0]=='-' or items[0][-1]=='-':
            continue
        if items[0][0]=='\'':
            continue
        if items[0][-1] in [str(i) for i in xrange(10)]:
            vocablst.add(items[0][:-1].lower())
        else:
            vocablst.add(items[0].lower())
vocablst.add('usage')
vocablst = sorted(list(vocablst))
with open('../exp/dict.lst', 'w') as writer:
    for word in vocablst:
        writer.write(word+'\n')
