"""convert id columns to UUID

Revision ID: d69b2716210f
Revises: 2fc0b07cbdf1
Create Date: 2026-02-12 16:24:39.146858

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'd69b2716210f'
down_revision: Union[str, Sequence[str], None] = '2fc0b07cbdf1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Step 1: Drop foreign key constraints
    op.drop_constraint('ocr_results_document_id_fkey', 'ocr_results', type_='foreignkey')
    op.drop_constraint('translations_document_id_fkey', 'translations', type_='foreignkey')

    # Step 2: Convert all columns from VARCHAR to UUID
    op.alter_column('documents', 'id',
               existing_type=sa.VARCHAR(),
               type_=postgresql.UUID(as_uuid=True),
               postgresql_using='id::uuid',
               existing_nullable=False)
    op.alter_column('ocr_results', 'id',
               existing_type=sa.VARCHAR(),
               type_=postgresql.UUID(as_uuid=True),
               postgresql_using='id::uuid',
               existing_nullable=False)
    op.alter_column('ocr_results', 'document_id',
               existing_type=sa.VARCHAR(),
               type_=postgresql.UUID(as_uuid=True),
               postgresql_using='document_id::uuid',
               existing_nullable=True)
    op.alter_column('translations', 'id',
               existing_type=sa.VARCHAR(),
               type_=postgresql.UUID(as_uuid=True),
               postgresql_using='id::uuid',
               existing_nullable=False)
    op.alter_column('translations', 'document_id',
               existing_type=sa.VARCHAR(),
               type_=postgresql.UUID(as_uuid=True),
               postgresql_using='document_id::uuid',
               existing_nullable=True)

    # Step 3: Re-create foreign key constraints
    op.create_foreign_key('ocr_results_document_id_fkey', 'ocr_results', 'documents', ['document_id'], ['id'])
    op.create_foreign_key('translations_document_id_fkey', 'translations', 'documents', ['document_id'], ['id'])


def downgrade() -> None:
    """Downgrade schema."""
    # Step 1: Drop foreign key constraints
    op.drop_constraint('ocr_results_document_id_fkey', 'ocr_results', type_='foreignkey')
    op.drop_constraint('translations_document_id_fkey', 'translations', type_='foreignkey')

    # Step 2: Convert all columns from UUID back to VARCHAR
    op.alter_column('translations', 'document_id',
               existing_type=postgresql.UUID(as_uuid=True),
               type_=sa.VARCHAR(),
               postgresql_using='document_id::text',
               existing_nullable=True)
    op.alter_column('translations', 'id',
               existing_type=postgresql.UUID(as_uuid=True),
               type_=sa.VARCHAR(),
               postgresql_using='id::text',
               existing_nullable=False)
    op.alter_column('ocr_results', 'document_id',
               existing_type=postgresql.UUID(as_uuid=True),
               type_=sa.VARCHAR(),
               postgresql_using='document_id::text',
               existing_nullable=True)
    op.alter_column('ocr_results', 'id',
               existing_type=postgresql.UUID(as_uuid=True),
               type_=sa.VARCHAR(),
               postgresql_using='id::text',
               existing_nullable=False)
    op.alter_column('documents', 'id',
               existing_type=postgresql.UUID(as_uuid=True),
               type_=sa.VARCHAR(),
               postgresql_using='id::text',
               existing_nullable=False)

    # Step 3: Re-create foreign key constraints
    op.create_foreign_key('ocr_results_document_id_fkey', 'ocr_results', 'documents', ['document_id'], ['id'])
    op.create_foreign_key('translations_document_id_fkey', 'translations', 'documents', ['document_id'], ['id'])
