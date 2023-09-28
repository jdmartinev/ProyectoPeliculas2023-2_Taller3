from django.shortcuts import render
from movie.models import Movie

import os
import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from dotenv import load_dotenv, find_dotenv

# Create your views here.
def recomendaciones(request):
    movies = Movie.objects.all()
    if request.method == 'GET':
        return render(request, 'recomendaciones.html', {
            'movies': movies
            })
    elif request.method == 'POST':
        searchMovie = request.POST.get('searchMovie')

        #Se lee del archivo .env la api key de openai
        _ = load_dotenv('../openAI.env')
        openai.api_key  = os.environ['openAI_api_key']

        req = searchMovie
        emb_req = get_embedding(req,engine='text-embedding-ada-002')

        sim = []
        for i in range(len(movies)):
            emb = movies[i].emb
            emb = list(np.frombuffer(emb))
            sim.append(cosine_similarity(emb,emb_req))
        sim = np.array(sim)
        idx = np.argmax(sim)
        idx = int(idx)

        movies = [movies[idx]]

        print(movies)

        return render(request, 'recomendaciones.html', {
            'movies': movies, 'searchTerm': searchMovie
            })
    