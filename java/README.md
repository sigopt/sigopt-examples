You will need maven to compile and run this example.
On OS X, you can install maven with Homebrew by running `brew install maven`.

# Compile

```bash
mvn compile
```

# Run:

Get your tokens from https://api.sigopt.com/user/profile, and then

```bash
mvn exec:java -Dexec.mainClass="com.example.App" -Dexec.args="--user_token USER_TOKEN --client_token CLIENT_TOKEN --client_id CLIENT_ID"
```
