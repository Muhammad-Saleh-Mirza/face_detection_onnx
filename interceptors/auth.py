#RULE: no need to change according to use cases
import grpc
import os

VALID_USERNAME = os.getenv("GRPC_USERNAME","verseye")
VALID_PASSWORD = os.getenv("GRPC_PASSWORD","verseye1@3")


class AuthInterceptor(grpc.ServerInterceptor):

    def intercept_service(self, continuation, handler_call_details):

        metadata = dict(handler_call_details.invocation_metadata)

        username = metadata.get("username")
        password = metadata.get("password")

        # Validate credentials
        if username != VALID_USERNAME or password != VALID_PASSWORD:

            def abort_handler(request, context):
                context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    "Invalid username or password"
                )

            return grpc.unary_unary_rpc_method_handler(
                abort_handler
            )

        return continuation(handler_call_details)