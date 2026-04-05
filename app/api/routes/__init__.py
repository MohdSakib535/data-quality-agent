from app.api.routes.analysis import router as analysis_router
from app.api.routes.chat import router as chat_router
from app.api.routes.clean import router as clean_router
from app.api.routes.files import router as files_router
from app.api.routes.system import router as system_router

__all__ = [
    "analysis_router",
    "chat_router",
    "clean_router",
    "files_router",
    "system_router",
]
