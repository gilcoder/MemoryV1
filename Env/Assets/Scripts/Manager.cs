using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using ai4u.ext;

public class Manager : MonoBehaviour
{

    public GameObject action1;
    public GameObject action2;
    public GameObject action3;
    public GameObject action4;
    public GameObject agent;
    public GameObject respawn1, respawn2, respawn3;

    void Awake() {
        ResetGame();
    }

    // Start is called before the first frame update
    public void ResetGame()
    {
        int choosePosition = Random.Range(0, 4);
        int chooseRespawn = Random.Range(0, 3);
        
        switch(chooseRespawn)
        {
            case 0:
                agent.transform.position = respawn1.transform.position;
                agent.transform.rotation = respawn1.transform.rotation;
                break;
            case 1:
                agent.transform.position = respawn2.transform.position;
                agent.transform.rotation = respawn2.transform.rotation;
                break;
            case 2:
                agent.transform.position = respawn3.transform.position;
                agent.transform.rotation = respawn3.transform.rotation;
                break;
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
