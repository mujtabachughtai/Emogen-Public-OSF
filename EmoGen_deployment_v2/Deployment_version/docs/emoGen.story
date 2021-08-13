#!/usr/bin/env python2

import meStory

# "meStory" is a python class that wraps up the process of taking 
# a markdown formatted document and using pandoc to output html,ebook
# or Word doc

# originally for my (Murray) story writing sideline, it can be nice for 
# throwing together documentation too.
# 
# All we have to do is:
# 1) instantiate a Story object
# 2) fill in the basic properties of the "Story"
# 3) write the "chapters" which are the individual markdown text files that make the bulk
#    of the document and tell the Story about them.
# 4) execute this file like any other Python source file.
st = meStory.Story()

st.style = "story2.css"

st.title    = "EmoGen"
st.author   = "Nadejda, Murray"
st.date     = [2021,8,4]
st.coverImg = "imgs/male.jpg"
#st.outputTypes = ["html", "word", "epub"]
st.outputTypes = ["html"]

st.MakeChapter( 0, "System overview", ["00.chp"] )
st.MakeChapter( 1, "EmoGen tool", ["01.chp", "01.code-review.chp"] )
st.MakeChapter( 2, "ApplyBlendweights tool", ["02.chp"])




st.Compile()
