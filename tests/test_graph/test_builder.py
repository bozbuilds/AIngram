# tests/test_graph/test_builder.py
import time

from aingram.graph.builder import GraphBuilder


class TestGraphBuilderUpsert:
    def test_upsert_creates_new_entity(self, engine):
        builder = GraphBuilder(engine)
        entity_id = builder.upsert_entity('Alice', 'person')
        assert isinstance(entity_id, str)

        entity = engine.get_entity(entity_id)
        assert entity.name == 'Alice'
        assert entity.entity_type == 'person'

    def test_upsert_existing_returns_same_id(self, engine):
        builder = GraphBuilder(engine)
        id1 = builder.upsert_entity('Alice', 'person')
        id2 = builder.upsert_entity('Alice', 'person')
        assert id1 == id2

    def test_upsert_existing_updates_last_seen(self, engine):
        builder = GraphBuilder(engine)
        entity_id = builder.upsert_entity('Alice', 'person')
        original = engine.get_entity(entity_id)

        time.sleep(0.01)
        builder.upsert_entity('Alice', 'person')
        updated = engine.get_entity(entity_id)
        assert updated.last_seen > original.last_seen

    def test_upsert_different_types_creates_separate(self, engine):
        builder = GraphBuilder(engine)
        id1 = builder.upsert_entity('Python', 'technology')
        id2 = builder.upsert_entity('Python', 'animal')
        assert id1 != id2


class TestGraphBuilderRelationships:
    def test_add_relationship(self, engine):
        builder = GraphBuilder(engine)
        src = builder.upsert_entity('Alice', 'person')
        tgt = builder.upsert_entity('Acme', 'organization')
        rel_id = builder.add_relationship(
            src,
            tgt,
            'works_at',
            fact='Alice works at Acme',
        )
        assert isinstance(rel_id, str)

    def test_get_entity_relationships(self, engine):
        builder = GraphBuilder(engine)
        src = builder.upsert_entity('Alice', 'person')
        tgt = builder.upsert_entity('Acme', 'organization')
        builder.add_relationship(src, tgt, 'works_at')

        rels = builder.get_entity_relationships(src)
        assert len(rels) == 1
        assert rels[0].relation_type == 'works_at'


class TestGraphBuilderQueries:
    def test_get_entities(self, engine):
        builder = GraphBuilder(engine)
        builder.upsert_entity('Alice', 'person')
        builder.upsert_entity('Acme', 'organization')

        entities = builder.get_entities()
        assert len(entities) == 2

    def test_get_entities_respects_limit(self, engine):
        builder = GraphBuilder(engine)
        for i in range(5):
            builder.upsert_entity(f'Entity{i}', 'test')

        entities = builder.get_entities(limit=3)
        assert len(entities) == 3
