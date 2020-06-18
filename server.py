from flask import Flask, render_template, request, Response, session
import json
import re

from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence
from .biunilm.question_generator import QuestionGenerator
from .vocabulary import Vocabulary
from transformers import pipeline

import torch
