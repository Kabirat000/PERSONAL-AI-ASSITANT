"""
Tests for the idea processing service.
"""
import json
import pytest
from unittest.mock import patch, MagicMock


class TestIdeaService:
    """Test cases for idea_service.py"""

    @patch("app.services.idea_service.run_llm_with_context")
    @patch("app.services.idea_service.retrieve_similar_ideas")
    @patch("app.services.idea_service.store_idea")
    def test_process_idea_success(self, mock_store, mock_retrieve, mock_llm):
        """Test successful idea processing."""
        from app.services.idea_service import process_idea

        # Mock return values
        mock_retrieve.return_value = ["related idea 1"]
        mock_llm.return_value = json.dumps({
            "clean_note": "Finish thesis and call mom",
            "themes": ["academic", "family"],
            "suggested_tasks": [
                {"task": "Complete thesis draft", "priority": "high"},
                {"task": "Call mom", "priority": "medium"},
            ],
        })
        mock_store.return_value = "doc-123"

        result = process_idea("need to finish thesis and also call mom")

        assert "clean_note" in result
        assert result["context_used"] is True
        assert result["related_ideas_count"] == 1
        mock_store.assert_called_once()

    @patch("app.services.idea_service.run_llm_with_context")
    @patch("app.services.idea_service.retrieve_similar_ideas")
    @patch("app.services.idea_service.store_idea")
    def test_process_idea_no_related(self, mock_store, mock_retrieve, mock_llm):
        """Test idea processing without related ideas."""
        from app.services.idea_service import process_idea

        mock_retrieve.return_value = []
        mock_llm.return_value = json.dumps({
            "clean_note": "New idea",
            "themes": ["general"],
            "suggested_tasks": [],
        })
        mock_store.return_value = "doc-456"

        result = process_idea("brand new idea")

        assert result["context_used"] is False
        assert result["related_ideas_count"] == 0

    @patch("app.services.idea_service.run_llm_with_context")
    @patch("app.services.idea_service.retrieve_similar_ideas")
    @patch("app.services.idea_service.store_idea")
    def test_process_idea_invalid_json(self, mock_store, mock_retrieve, mock_llm):
        """Test handling of invalid JSON from LLM."""
        from app.services.idea_service import process_idea

        mock_retrieve.return_value = []
        mock_llm.return_value = "This is not valid JSON"
        mock_store.return_value = "doc-789"

        result = process_idea("some idea")

        assert "error" in result
        assert "raw_output" in result


class TestTaskService:
    """Test cases for task_service.py"""

    @patch("app.services.task_service.run_llm_structured")
    def test_extract_tasks_success(self, mock_llm):
        """Test successful task extraction."""
        from app.services.task_service import extract_tasks

        mock_llm.return_value = json.dumps({
            "tasks": [
                {"task": "Buy groceries", "priority": "medium"},
                {"task": "Call dentist", "priority": "high"},
            ]
        })

        tasks = extract_tasks("need to buy groceries and call the dentist")

        assert len(tasks) == 2
        assert tasks[0]["task"] == "Buy groceries"
        assert tasks[1]["priority"] == "high"

    @patch("app.services.task_service.run_llm_structured")
    def test_extract_tasks_empty(self, mock_llm):
        """Test extraction when no tasks found."""
        from app.services.task_service import extract_tasks

        mock_llm.return_value = json.dumps({"tasks": []})

        tasks = extract_tasks("I wonder what the weather is like")

        assert tasks == []


class TestEmbeddingService:
    """Test cases for embedding_service.py"""

    @patch("app.services.embedding_service.get_settings")
    @patch("app.services.embedding_service.OpenAI")
    def test_openai_provider_initialization(self, mock_openai, mock_settings):
        """Test OpenAI provider initializes correctly."""
        mock_settings.return_value = MagicMock(
            openai_api_key="test-key",
            openai_embedding_model="text-embedding-3-small",
            openai_embedding_dim=1536,
            embedding_provider="openai",
        )

        from app.services.embedding_service import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider()
        assert provider.dimension == 1536


class TestQdrantStore:
    """Test cases for qdrant_store.py"""

    @patch("app.rag.qdrant_store.QdrantClient")
    @patch("app.rag.qdrant_store.get_settings")
    def test_ensure_collection_creates_new(self, mock_settings, mock_client):
        """Test collection creation when it doesn't exist."""
        mock_settings.return_value = MagicMock(
            qdrant_host="localhost",
            qdrant_port=6333,
            qdrant_api_key="",
            qdrant_collection_name="test",
            active_embedding_dim=1536,
        )

        mock_instance = MagicMock()
        mock_instance.get_collections.return_value = MagicMock(collections=[])
        mock_client.return_value = mock_instance

        from app.rag.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore(collection_name="test", embedding_dim=1536)

        mock_instance.create_collection.assert_called_once()

    @patch("app.rag.qdrant_store.QdrantClient")
    @patch("app.rag.qdrant_store.get_settings")
    def test_ensure_collection_preserves_existing(self, mock_settings, mock_client):
        """Test that existing collection is not recreated."""
        mock_settings.return_value = MagicMock(
            qdrant_host="localhost",
            qdrant_port=6333,
            qdrant_api_key="",
            qdrant_collection_name="test",
            active_embedding_dim=1536,
        )

        mock_instance = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "test"
        mock_instance.get_collections.return_value = MagicMock(collections=[mock_collection])
        mock_client.return_value = mock_instance

        from app.rag.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore(collection_name="test", embedding_dim=1536)

        mock_instance.create_collection.assert_not_called()
