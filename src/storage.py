from __future__ import annotations

import os
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from src.config import logger

from pydantic_settings import BaseSettings
import asyncio
from dotenv import load_dotenv



class Settings(BaseSettings):
    """Base settings object to inherit from."""

    class Config:
        env_file = os.path.join(os.getcwd(), ".env")
        env_file_encoding = "utf-8"


HOSTS = os.getenv('ES_HOSTS')
LOGIN = os.getenv('ES_LOGIN')
PASSWORD = os.getenv('ES_PASSWORD')


class ElasticClient(AsyncElasticsearch):
    """Handling with AsyncElasticsearch"""
    def __init__(self, *args, **kwargs):
        self.max_hits = 300
        self.chunk_size = 500
        self.loop = asyncio.new_event_loop()
        super().__init__(
            hosts=HOSTS,
            basic_auth=(LOGIN, PASSWORD),
            request_timeout=100,
            max_retries=50,
            retry_on_timeout=True,
            *args,
            **kwargs,
        )


    def create_index(self, index_name: str = None) -> None:
        """
        :param index_name:
        """

        async def create(index: str = None) -> None:
            """Creates the index if one does not exist."""
            try:
                await self.indices.create(index=index)
                await self.close()
            except:
                await self.close()
                logger.info("impossible create index with name {}".format(index_name))

        self.loop.run_until_complete(create(index_name))


    def delete_index(self, index_name) -> None:
        """Deletes the index if one exists."""

        async def delete(index: str):
            """
            :param index:
            """
            try:
                await self.indices.delete(index=index)
                await self.close()
            except:
                await self.close()
                logger.info("impossible delete index with name {}".format(index_name))

        self.loop.run_until_complete(delete(index_name))

    def search_query(self, index: str, query: dict):
        
        async def search_by_query(index_: str, query_: dict):
            """
            :param query:
            :return:
            """
            resp = await self.search(
                allow_partial_search_results=True,
                min_score=0,
                index=index_,
                query=query_,
                size=self.max_hits)
            
            return resp
            # await self.close()

        return self.loop.run_until_complete(search_by_query(index, query))

    def add_docs(self, index_name: str, docs: list[dict]):
        async def add_docs_bulk(index_name_: str, docs_: list[dict]):
            """
            :param index_name:
            :param docs:
            """
            try:
                _gen = ({"_index": index_name_, "_source": doc} for doc in docs_)
                await async_bulk(
                    self, _gen, chunk_size=self.chunk_size, stats_only=True
                )
                logger.info("adding {} documents in index {}".format(len(docs_), index_name_))
            except Exception:
                logger.exception(
                    "Impossible adding {} documents in index {}".format(
                        len(docs_), index_name_
                    )
                )
        self.loop.run_until_complete(add_docs_bulk(index_name, docs))
        