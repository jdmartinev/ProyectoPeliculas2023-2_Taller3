from django.shortcuts import render
from movie.models import Movie

from dotenv import load_dotenv
import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import os

_ = load_dotenv('../openAI.env')
openai.api_key  = os.environ['openAI_api_key']

def recomendaciones(request):
    rec = request.GET.get('askRec')
    if rec:
        emb = get_embedding(rec, engine='text-embedding-ada-002')
        movies = list(Movie.objects.all())
        sim = []
        for movie in movies:
            emb_binary = np.array(movie.emb).tobytes()
            rec_emb = list(np.frombuffer(emb_binary))
            sim.append(cosine_similarity(emb, rec_emb))
        sim = np.array(sim)
        idx = np.argmax(sim)

        recommended_movie = movies[idx]

        return render(request, 'reco.html', {'movie': recommended_movie})

    else:
        return render(request, 'reco.html')