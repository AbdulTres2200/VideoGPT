import dropbox

dbx = dropbox.Dropbox(
    oauth2_refresh_token="KzDCCHHbYaoAAAAAAAAAAeTMAGOmvNK91deQlAtgQZia4agFZlP0Uk0XgMuBjqmM",
    app_key="vjbmnbvv6sm9l5x",
    app_secret="sxrqiv4b867u34t"
)

# SDK auto-refreshes the access token on every API call
print(dbx.users_get_current_account())

# import dropbox

# auth_flow = dropbox.DropboxOAuth2FlowNoRedirect(
#     consumer_key="vjbmnbvv6sm9l5x",
#     consumer_secret="sxrqiv4b867u34t",
#     token_access_type='offline'  # <<< gives refresh token
# )

# authorize_url = auth_flow.start()
# print("1. Go to: ", authorize_url)
# auth_code = input("2. Enter the authorization code here: ")

# oauth_result = auth_flow.finish(auth_code)
# print("REFRESH TOKEN:", oauth_result.refresh_token)
