from app.database import Base


class Highlight(Base):
    __tablename__ = 'highlights'

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    book_id = Column(Integer, ForeignKey('books.id'))
    book = relationship("Book", back_populates="highlights")