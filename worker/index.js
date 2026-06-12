import { Container, getContainer } from "@cloudflare/containers";

/**
 * Durable Object wrapping the FastAPI container.
 * The container image is built from ./Dockerfile and listens on port 8000.
 */
export class JobRagContainer extends Container {
  defaultPort = 8000;
  sleepAfter = "15m";

  constructor(ctx, env) {
    super(ctx, env);
    // Forward Worker secrets/vars into the container environment.
    this.envVars = {
      GROQ_API_KEY: env.GROQ_API_KEY ?? "",
      GOOGLE_API_KEY: env.GOOGLE_API_KEY ?? "",
      CHROMA_DIR: "/app/data/chroma",
    };
  }
}

export default {
  async fetch(request, env) {
    // Route every request to the (single) container instance.
    return getContainer(env.JOB_RAG).fetch(request);
  },
};
